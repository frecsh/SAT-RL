import gzip
from enum import Enum
from typing import Any

import lz4.frame

try:
    import zstandard
except ImportError:
    zstandard = None


class CompressionFormat(Enum):
    """Enum for supported compression formats."""

    NONE = "none"
    ZSTD = "zstd"
    GZIP = "gzip"
    LZ4 = "lz4"


class CompressionLevel(Enum):
    """Predefined compression levels."""

    NONE = 0
    FAST = 1
    BALANCED = 5
    HIGH = 9


class CompressionConfig:
    """Configuration for compression settings."""

    def __init__(
        self,
        format: CompressionFormat | str = CompressionFormat.NONE,
        level: CompressionLevel | int = CompressionLevel.BALANCED,
    ):
        """
        Initialize compression configuration.

        Args:
            format: Compression format to use
            level: Compression level (0-9, where 0 is no compression)
        """
        if isinstance(format, str):
            format = CompressionFormat(format.lower())
        if isinstance(level, int):
            level = level
        else:
            level = level.value

        self.format = format
        self.level = level

    @property
    def is_enabled(self) -> bool:
        """Check if compression is enabled."""
        return self.format != CompressionFormat.NONE and self.level > 0


class FieldCompressionManager:
    """Manager for field-specific compression configurations."""

    def __init__(self, default_config: CompressionConfig | None = None):
        """
        Initialize the field compression manager.

        Args:
            default_config: Default compression config for fields without specific settings
        """
        self.default_config = default_config or CompressionConfig()
        self.field_configs: dict[str, CompressionConfig] = {}

    def set_field_config(self, field_name: str, config: CompressionConfig):
        """Set compression configuration for a specific field."""
        self.field_configs[field_name] = config

    def get_field_config(self, field_name: str) -> CompressionConfig:
        """Get compression configuration for a specific field."""
        return self.field_configs.get(field_name, self.default_config)

    def should_compress_field(self, field_name: str) -> bool:
        """Check if a field should be compressed."""
        return self.get_field_config(field_name).is_enabled


class CompressionService:
    """Service for handling data compression and decompression."""

    @staticmethod
    def compress(
        data: bytes, config: CompressionConfig
    ) -> tuple[bytes, dict[str, Any]]:
        """
        Compress data according to the provided configuration.

        Args:
            data: Raw data to compress
            config: Compression configuration

        Returns:
            Tuple of (compressed_data, metadata)
        """
        if not config.is_enabled:
            return data, {"compression": "none"}

        metadata = {"compression": config.format.value, "level": config.level}

        if config.format == CompressionFormat.ZSTD:
            return compress_zstd(data, config.level)
        elif config.format == CompressionFormat.GZIP:
            return gzip.compress(data, compresslevel=config.level), metadata
        elif config.format == CompressionFormat.LZ4:
            return lz4.frame.compress(data, compression_level=config.level), metadata
        else:
            return data, {"compression": "none"}

    @staticmethod
    def decompress(data: bytes, metadata: dict[str, Any]) -> bytes:
        """
        Decompress data according to the provided metadata.

        Args:
            data: Compressed data
            metadata: Metadata containing compression information

        Returns:
            Decompressed data
        """
        compression_format = metadata.get("compression", "none")

        if compression_format == "none" or not compression_format:
            return data
        elif compression_format == "zstd":
            return decompress_zstd(data, metadata)
        elif compression_format == "gzip":
            return gzip.decompress(data)
        elif compression_format == "lz4":
            return lz4.frame.decompress(data)
        else:
            raise ValueError(f"Unsupported compression format: {compression_format}")


def compress_zstd(data: bytes, level: int = 3) -> tuple[bytes, dict[str, Any]]:
    """Compress data using zstd compression.

    Args:
        data: Data to compress
        level: Compression level (1-22, higher means more compression)

    Returns:
        Tuple of (compressed_data, metadata)
    """
    if zstandard is None:
        raise ImportError("zstandard package is required for zstd compression")

    # Create a compressor with the specified level
    compressor = zstandard.ZstdCompressor(level=level)

    # Compress the data
    compressed = compressor.compress(data)

    # Return the compressed data and metadata
    metadata = {"compression": "zstd", "level": level, "original_size": len(data)}
    return compressed, metadata


def decompress_zstd(data: bytes, metadata: dict[str, Any]) -> bytes:
    """Decompress data using zstd compression.

    Args:
        data: Compressed data
        metadata: Compression metadata

    Returns:
        Decompressed data
    """
    if zstandard is None:
        raise ImportError("zstandard package is required for zstd decompression")

    # Create a decompressor
    decompressor = zstandard.ZstdDecompressor()

    # Decompress the data and return
    return decompressor.decompress(data)
