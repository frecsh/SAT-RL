import unittest
import os
import numpy as np
import tempfile
import yaml
import sys
from pathlib import Path
import shutil

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.compression import (
    CompressionFormat, 
    CompressionLevel, 
    CompressionConfig, 
    FieldCompressionManager,
    CompressionService
)
from experience.storage import (
    create_storage, 
    ParquetExperienceStorage,
    HDF5ExperienceStorage,
    MemoryMappedExperienceStorage
)


class TestCompression(unittest.TestCase):
    """Test case for compression utilities."""
    
    def test_compression_config(self):
        """Test creating and using compression configuration."""
        config = CompressionConfig(CompressionFormat.ZSTD, CompressionLevel.HIGH)
        self.assertEqual(config.format, CompressionFormat.ZSTD)
        self.assertEqual(config.level, 9)
        self.assertTrue(config.is_enabled)
        
        # Test string format constructor
        config = CompressionConfig("gzip", 3)
        self.assertEqual(config.format, CompressionFormat.GZIP)
        self.assertEqual(config.level, 3)
        self.assertTrue(config.is_enabled)
        
        # Test none format
        config = CompressionConfig(CompressionFormat.NONE, CompressionLevel.HIGH)
        self.assertFalse(config.is_enabled)
    
    def test_field_compression_manager(self):
        """Test field compression manager functionality."""
        manager = FieldCompressionManager()
        # Default should be none/disabled
        self.assertFalse(manager.default_config.is_enabled)
        self.assertFalse(manager.should_compress_field("test_field"))
        
        # Set default config
        manager.default_config = CompressionConfig(CompressionFormat.ZSTD, CompressionLevel.BALANCED)
        self.assertTrue(manager.should_compress_field("test_field"))
        
        # Set field-specific config
        manager.set_field_config(
            "specific_field", 
            CompressionConfig(CompressionFormat.LZ4, CompressionLevel.FAST)
        )
        self.assertEqual(manager.get_field_config("specific_field").format, CompressionFormat.LZ4)
        self.assertEqual(manager.get_field_config("specific_field").level, 1)
        
        # Set field to not be compressed
        manager.set_field_config(
            "uncompressed_field",
            CompressionConfig(CompressionFormat.NONE, 0)
        )
        self.assertFalse(manager.should_compress_field("uncompressed_field"))
    
    def test_compression_service(self):
        """Test compression and decompression with different formats."""
        test_data = b"This is test data that should be compressed and then decompressed correctly."
        
        for format_name in ["zstd", "gzip", "lz4"]:
            config = CompressionConfig(format_name, 5)
            compressed, metadata = CompressionService.compress(test_data, config)
            
            # Compressed data should be different from original
            self.assertNotEqual(compressed, test_data)
            
            # Metadata should contain format and level
            self.assertEqual(metadata["compression"], format_name)
            self.assertEqual(metadata["level"], 5)
            
            # Decompress should return original data
            decompressed = CompressionService.decompress(compressed, metadata)
            self.assertEqual(decompressed, test_data)
        
        # Test no compression
        config = CompressionConfig(CompressionFormat.NONE, 0)
        compressed, metadata = CompressionService.compress(test_data, config)
        self.assertEqual(compressed, test_data)
        self.assertEqual(metadata["compression"], "none")


class TestStorageWithCompression(unittest.TestCase):
    """Test case for storage backends with compression support."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = {
            "observations": np.random.rand(100, 50).astype(np.float32),
            "actions": np.random.randint(0, 10, size=(100,), dtype=np.int32),
            "rewards": np.random.rand(100).astype(np.float32),
            "masks": np.ones((100, 10), dtype=np.float32),
            "timestamps": np.arange(100, dtype=np.int64)
        }
        
        # Create a compression configuration similar to the YAML file
        self.compression_config = {
            "default": {
                "format": "zstd",
                "level": 3
            },
            "fields": {
                "observations": {
                    "format": "zstd",
                    "level": 7
                },
                "rewards": {
                    "format": "lz4",
                    "level": 1
                },
                "actions": {
                    "format": "zstd",
                    "level": 3
                },
                "masks": {
                    "format": "gzip",
                    "level": 5
                },
                "timestamps": {
                    "format": "none",
                    "level": 0
                }
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
    
    def test_parquet_storage_with_compression(self):
        """Test Parquet storage with compression."""
        path = os.path.join(self.test_dir, "test_parquet.parquet")
        
        # Create storage and write data
        storage = create_storage("parquet", path, self.compression_config)
        storage.write_batch(self.test_data, {"test_meta": "value"})
        
        # Read data back
        read_data = storage.read_batch()
        
        # Verify all data matches the original
        for key, value in self.test_data.items():
            np.testing.assert_array_equal(read_data[key], value)
    
    def test_hdf5_storage_with_compression(self):
        """Test HDF5 storage with compression."""
        path = os.path.join(self.test_dir, "test_hdf5.h5")
        
        # Create storage and write data
        storage = create_storage("hdf5", path, self.compression_config)
        storage.write_batch(self.test_data, {"test_meta": "value"})
        
        # Read data back
        read_data = storage.read_batch()
        
        # Verify all data matches the original
        for key, value in self.test_data.items():
            np.testing.assert_array_equal(read_data[key], value)
    
    def test_mmap_storage_with_compression(self):
        """Test memory-mapped storage with compression."""
        path = os.path.join(self.test_dir, "test_mmap.dat")
        
        # Create storage and write data
        storage = create_storage("mmap", path, self.compression_config)
        storage.write_batch(self.test_data, {"test_meta": "value"})
        
        # Read data back
        read_data = storage.read_batch()
        
        # Verify all data matches the original
        for key, value in self.test_data.items():
            np.testing.assert_array_equal(read_data[key], value)
    
    def test_partial_field_reading(self):
        """Test reading only specific fields."""
        path = os.path.join(self.test_dir, "test_partial.parquet")
        
        # Create storage and write data
        storage = create_storage("parquet", path, self.compression_config)
        storage.write_batch(self.test_data, {"test_meta": "value"})
        
        # Read only specific fields
        fields_to_read = ["observations", "rewards"]
        read_data = storage.read_batch(fields_to_read)
        
        # Verify only requested fields are returned
        self.assertEqual(set(read_data.keys()), set(fields_to_read))
        
        # Verify data matches the original
        for key in fields_to_read:
            np.testing.assert_array_equal(read_data[key], self.test_data[key])
    
    def test_config_loading_from_yaml(self):
        """Test loading compression config from YAML file."""
        # Create a sample YAML file
        yaml_path = os.path.join(self.test_dir, "test_config.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(self.compression_config, f)
        
        # Load the config
        with open(yaml_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Create storage with loaded config
        storage_path = os.path.join(self.test_dir, "yaml_config_test.parquet")
        storage = create_storage("parquet", storage_path, loaded_config)
        
        # Write and read data
        storage.write_batch(self.test_data)
        read_data = storage.read_batch()
        
        # Verify data matches
        for key, value in self.test_data.items():
            np.testing.assert_array_equal(read_data[key], value)


class TestCompressionPerformance(unittest.TestCase):
    """Test performance characteristics of different compression strategies."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create large test data
        self.large_data = {
            "large_observations": np.random.rand(1000, 500).astype(np.float32),
            "sparse_data": np.zeros((2000, 1000), dtype=np.float32)
        }
        # Add some sparse data with a few non-zero elements
        for i in range(2000):
            self.large_data["sparse_data"][i, np.random.randint(0, 1000, 5)] = 1.0
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)
    
    def test_compression_ratio_comparison(self):
        """Compare compression ratios across different formats."""
        results = {}
        
        # Test different compression formats
        for format_name in ["none", "zstd", "gzip", "lz4"]:
            config = {
                "default": {
                    "format": format_name,
                    "level": 5 if format_name != "none" else 0
                }
            }
            
            path = os.path.join(self.test_dir, f"test_{format_name}.parquet")
            
            # Create storage and write data
            storage = create_storage("parquet", path, config)
            storage.write_batch(self.large_data)
            
            # Get file size
            size = os.path.getsize(path)
            results[format_name] = size
        
        # Verify that compressed formats are smaller than uncompressed
        self.assertLess(results["zstd"], results["none"])
        self.assertLess(results["gzip"], results["none"])
        self.assertLess(results["lz4"], results["none"])
        
        # Print results for debugging (commented out for tests)
        # for format_name, size in results.items():
        #     print(f"{format_name}: {size / 1024:.2f} KB")


if __name__ == "__main__":
    unittest.main()