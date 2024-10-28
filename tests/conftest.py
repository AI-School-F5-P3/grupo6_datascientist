import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_streamlit():
    """Fixture que proporciona un mock de streamlit"""
    return MagicMock()

@pytest.fixture
def mock_models():
    """Fixture que proporciona un mock de los modelos"""
    return {
        'vision': MagicMock()
    }