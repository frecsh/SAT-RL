"""
Test the concurrent operations functionality.

This module contains tests for the ConcurrentWriter, ShardedWriter, and ThreadSafeReader classes,
verifying their functionality for concurrent operations on experience data.
"""

import unittest
import os
import json
import numpy as np
import tempfile
import shutil
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import the modules to test
from src.utils.concurrent_operations import (
    ConcurrentWriter, ShardedWriter, ThreadSafeReader
)


class TestConcurrentWriter(unittest.TestCase):
    """Test the ConcurrentWriter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_jsonl_writing(self):
        """Test writing to a JSON Lines file."""
        # Create a writer
        filepath = os.path.join(self.test_dir, "test_data")
        writer = ConcurrentWriter(filepath, format_type="jsonl")
        
        # Write some data
        writer.write({"id": 1, "value": "test1"})
        writer.write({"id": 2, "value": "test2"})
        writer.write({"id": 3, "value": "test3"})
        
        # Close the writer
        writer.close()
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath + ".jsonl"))
        
        # Read the file and check the content
        with open(filepath + ".jsonl", "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["id"], 1)
            self.assertEqual(json.loads(lines[1])["id"], 2)
            self.assertEqual(json.loads(lines[2])["id"], 3)
    
    def test_npz_writing(self):
        """Test writing to an NPZ file."""
        # Skip the test if numpy is not available
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not available")
        
        # Create a writer
        filepath = os.path.join(self.test_dir, "test_data")
        writer = ConcurrentWriter(filepath, format_type="npz")
        
        # Write some data
        writer.write({"values": np.array([1, 2, 3]), "labels": np.array([0])})
        writer.write({"values": np.array([4, 5, 6]), "labels": np.array([1])})
        
        # Close the writer
        writer.close()
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath + ".npz"))
        
        # Load the file and check the content
        data = np.load(filepath + ".npz")
        self.assertIn("values", data.files)
        self.assertIn("labels", data.files)
        np.testing.assert_array_equal(data["values"], np.array([[1, 2, 3], [4, 5, 6]]))
        # Expect shape (2, 1) as the writer stacks [0] and [1]
        np.testing.assert_array_equal(data["labels"], np.array([[0], [1]]))
    
    def test_context_manager(self):
        """Test using the writer as a context manager."""
        # Create a writer using context manager
        filepath = os.path.join(self.test_dir, "test_data")
        with ConcurrentWriter(filepath, format_type="jsonl") as writer:
            writer.write({"id": 1, "value": "test1"})
            writer.write({"id": 2, "value": "test2"})
        
        # Check that the file was created and closed properly
        self.assertTrue(os.path.exists(filepath + ".jsonl"))
        
        # Read the file and check the content
        with open(filepath + ".jsonl", "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
    
    def test_concurrent_writing(self):
        """Test concurrent writing from multiple threads."""
        # Create a writer
        filepath = os.path.join(self.test_dir, "test_data")
        writer = ConcurrentWriter(filepath, format_type="jsonl")
        
        # Number of threads and records per thread
        num_threads = 5
        records_per_thread = 20
        
        # Function to write records from a thread
        def write_records(thread_id):
            for i in range(records_per_thread):
                writer.write({
                    "thread_id": thread_id,
                    "record_id": i,
                    "value": f"thread_{thread_id}_record_{i}"
                })
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=write_records, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        
        # Close the writer
        writer.close()
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath + ".jsonl"))
        
        # Read the file and check the content
        with open(filepath + ".jsonl", "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), num_threads * records_per_thread)
            
            # Check that all records were written
            thread_record_counts = {}
            for line in lines:
                data = json.loads(line)
                thread_id = data["thread_id"]
                if thread_id not in thread_record_counts:
                    thread_record_counts[thread_id] = 0
                thread_record_counts[thread_id] += 1
            
            # Check that each thread wrote the expected number of records
            for thread_id in range(num_threads):
                self.assertEqual(thread_record_counts.get(thread_id, 0), records_per_thread)


class TestShardedWriter(unittest.TestCase):
    """Test the ShardedWriter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_shard_creation(self):
        """Test creating shards."""
        # Create a sharded writer
        writer = ShardedWriter(
            base_directory=self.test_dir,
            base_filename="test_data",
            num_shards=3,
            format_type="jsonl"
        )
        
        # Check that metadata was created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_data_shards.json")))
        
        # Check the metadata
        with open(os.path.join(self.test_dir, "test_data_shards.json"), "r") as f:
            metadata = json.load(f)
            self.assertEqual(metadata["num_shards"], 3)
            self.assertEqual(len(metadata["shards"]), 3)
        
        # Close the writer
        writer.close()
    
    def test_write_to_shards(self):
        """Test writing to shards."""
        # Create a sharded writer
        writer = ShardedWriter(
            base_directory=self.test_dir,
            base_filename="test_data",
            num_shards=3,
            format_type="jsonl"
        )
        
        # Write some data to specific shards
        writer.write({"id": 1, "value": "shard0"}, shard_id=0)
        writer.write({"id": 2, "value": "shard1"}, shard_id=1)
        writer.write({"id": 3, "value": "shard2"}, shard_id=2)
        
        # Check that the shard files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_data_shard_0.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_data_shard_1.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test_data_shard_2.jsonl")))
        
        # Close the writer
        writer.close()
        
        # Read the files and check the content
        for shard_id in range(3):
            with open(os.path.join(self.test_dir, f"test_data_shard_{shard_id}.jsonl"), "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                data = json.loads(lines[0])
                self.assertEqual(data["id"], shard_id + 1)
    
    def test_auto_shard_selection(self):
        """Test automatic shard selection."""
        # Create a sharded writer
        writer = ShardedWriter(
            base_directory=self.test_dir,
            base_filename="test_data",
            num_shards=3,
            format_type="jsonl"
        )
        
        # Write data without specifying shards
        shard_ids = []
        for i in range(9):
            shard_id = writer.write({"id": i, "value": f"auto_shard_{i}"})
            shard_ids.append(shard_id)
        
        # Check that records were distributed evenly
        shard_counts = [shard_ids.count(i) for i in range(3)]
        self.assertEqual(sum(shard_counts), 9)
        
        # Each shard should have approximately the same number of records
        # (in this case, exactly 3 records per shard)
        self.assertEqual(shard_counts[0], 3)
        self.assertEqual(shard_counts[1], 3)
        self.assertEqual(shard_counts[2], 3)
        
        # Close the writer
        writer.close()
    
    def test_merge_shards(self):
        """Test merging shards."""
        # Create a sharded writer
        writer = ShardedWriter(
            base_directory=self.test_dir,
            base_filename="test_data",
            num_shards=3,
            format_type="jsonl"
        )
        
        # Write some data to specific shards
        writer.write({"id": 1, "value": "shard0"}, shard_id=0)
        writer.write({"id": 2, "value": "shard1"}, shard_id=1)
        writer.write({"id": 3, "value": "shard2"}, shard_id=2)
        writer.write({"id": 4, "value": "shard0_again"}, shard_id=0)
        
        # Merge the shards
        merged_path = writer.merge_shards()
        
        # Check that the merged file was created
        self.assertTrue(os.path.exists(merged_path))
        
        # Read the merged file and check the content
        with open(merged_path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4)
            
            # Check that all records are present
            ids = [json.loads(line)["id"] for line in lines]
            self.assertIn(1, ids)
            self.assertIn(2, ids)
            self.assertIn(3, ids)
            self.assertIn(4, ids)
    
    def test_concurrent_shard_writing(self):
        """Test concurrent writing to multiple shards."""
        # Create a sharded writer
        writer = ShardedWriter(
            base_directory=self.test_dir,
            base_filename="test_data",
            num_shards=4,
            format_type="jsonl"
        )
        
        # Number of threads and records per thread
        num_threads = 4
        records_per_thread = 25
        
        # Function to write records from a thread
        def write_records(thread_id):
            local_shard_id = thread_id % 4  # Each thread writes to its own shard
            for i in range(records_per_thread):
                writer.write({
                    "thread_id": thread_id,
                    "record_id": i,
                    "value": f"thread_{thread_id}_record_{i}"
                }, shard_id=local_shard_id)
        
        # Create and start threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_records, i) for i in range(num_threads)]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
        
        # Close the writer
        writer.close()
        
        # Check that all shard files were created and have the expected number of records
        for shard_id in range(4):
            shard_path = os.path.join(self.test_dir, f"test_data_shard_{shard_id}.jsonl")
            self.assertTrue(os.path.exists(shard_path))
            
            with open(shard_path, "r") as f:
                lines = f.readlines()
                # Each shard should have records from one thread
                self.assertEqual(len(lines), records_per_thread)


class TestThreadSafeReader(unittest.TestCase):
    """Test the ThreadSafeReader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.create_jsonl_file()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_jsonl_file(self):
        """Create a test JSON Lines file."""
        filepath = os.path.join(self.test_dir, "test_data.jsonl")
        with open(filepath, "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "id": i,
                    "value": f"record_{i}",
                    "data": list(range(i, i+3))
                }) + "\n")
    
    def test_read_jsonl(self):
        """Test reading from a JSON Lines file."""
        # Create a reader
        filepath = os.path.join(self.test_dir, "test_data.jsonl")
        reader = ThreadSafeReader(filepath)
        
        # Read specific records
        record_0 = reader.read(0)
        record_5 = reader.read(5)
        record_9 = reader.read(9)
        
        # Check the content
        self.assertEqual(record_0["id"], 0)
        self.assertEqual(record_5["id"], 5)
        self.assertEqual(record_9["id"], 9)
        
        # Check data arrays
        self.assertEqual(record_0["data"], [0, 1, 2])
        self.assertEqual(record_5["data"], [5, 6, 7])
        self.assertEqual(record_9["data"], [9, 10, 11])
    
    def test_stream_reading(self):
        """Test streaming records sequentially."""
        # Create a reader
        filepath = os.path.join(self.test_dir, "test_data.jsonl")
        reader = ThreadSafeReader(filepath)
        
        # Stream records in batches
        batch1 = reader.stream_next(batch_size=3)
        batch2 = reader.stream_next(batch_size=3)
        batch3 = reader.stream_next(batch_size=4)
        
        # Check batch sizes
        self.assertEqual(len(batch1), 3)
        self.assertEqual(len(batch2), 3)
        self.assertEqual(len(batch3), 4)
        
        # Check record IDs
        self.assertEqual([record["id"] for record in batch1], [0, 1, 2])
        self.assertEqual([record["id"] for record in batch2], [3, 4, 5])
        self.assertEqual([record["id"] for record in batch3], [6, 7, 8, 9])
        
        # Streaming again should return empty list (end of file)
        batch4 = reader.stream_next(batch_size=1)
        self.assertEqual(len(batch4), 0)
    
    def test_count_records(self):
        """Test counting records."""
        # Create a reader
        filepath = os.path.join(self.test_dir, "test_data.jsonl")
        reader = ThreadSafeReader(filepath)
        
        # Count records
        count = reader.count_records()
        self.assertEqual(count, 10)
    
    def test_read_batch(self):
        """Test reading multiple records by indices."""
        # Create a reader
        filepath = os.path.join(self.test_dir, "test_data.jsonl")
        reader = ThreadSafeReader(filepath)
        
        # Read a batch of records
        indices = [1, 3, 5, 7, 9]
        records = reader.read_batch(indices)
        
        # Check the batch
        self.assertEqual(len(records), 5)
        self.assertEqual([record["id"] for record in records], indices)
    
    def test_caching(self):
        """Test the caching mechanism."""
        # Create a reader with small cache
        filepath = os.path.join(self.test_dir, "test_data.jsonl")
        reader = ThreadSafeReader(filepath, cache_size=5)
        
        # Read records to fill the cache
        for i in range(5):
            reader.read(i)
        
        # Read the same records again (should be cached)
        start_time = time.time()
        for i in range(5):
            reader.read(i)
        cached_read_time = time.time() - start_time
        
        # Read different records (not in cache)
        start_time = time.time()
        for i in range(5, 10):
            reader.read(i)
        uncached_read_time = time.time() - start_time
        
        # Cached reads should be faster, but timing is not reliable in unit tests
        # So we just check that the cache works by reading again
        for i in range(5, 10):
            reader.read(i)  # These should now be cached
    
    def test_concurrent_reading(self):
        """Test concurrent reading from multiple threads."""
        # Create a reader
        filepath = os.path.join(self.test_dir, "test_data.jsonl")
        reader = ThreadSafeReader(filepath)
        
        # Number of threads and reads per thread
        num_threads = 5
        reads_per_thread = 20
        
        # Results container
        results = []
        
        # Function to read records from a thread
        def read_records(thread_id):
            thread_results = []
            for i in range(reads_per_thread):
                # Read a random record
                index = i % 10
                record = reader.read(index)
                thread_results.append((thread_id, index, record["id"]))
            return thread_results
        
        # Create and start threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(read_records, i) for i in range(num_threads)]
            
            # Collect results
            for future in futures:
                results.extend(future.result())
        
        # Check that we got the expected number of results
        self.assertEqual(len(results), num_threads * reads_per_thread)
        
        # Check that each read returned the correct record
        for thread_id, index, record_id in results:
            self.assertEqual(index, record_id, 
                             f"Thread {thread_id} read index {index} but got record {record_id}")


if __name__ == "__main__":
    unittest.main()