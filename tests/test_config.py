import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import config


class TestConfig:
    """Tests for configuration validation"""

    def test_max_results_is_positive(self):
        """MAX_RESULTS must be positive for search to work"""
        assert config.MAX_RESULTS > 0, "MAX_RESULTS must be greater than 0"

    def test_chunk_size_is_positive(self):
        """CHUNK_SIZE must be positive"""
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be greater than 0"

    def test_chunk_overlap_less_than_chunk_size(self):
        """CHUNK_OVERLAP must be less than CHUNK_SIZE"""
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, \
            "CHUNK_OVERLAP must be less than CHUNK_SIZE"

    def test_max_history_is_non_negative(self):
        """MAX_HISTORY must be non-negative"""
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY must be >= 0"

    def test_anthropic_model_is_set(self):
        """ANTHROPIC_MODEL must be set"""
        assert config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL must be set"

    def test_embedding_model_is_set(self):
        """EMBEDDING_MODEL must be set"""
        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL must be set"
