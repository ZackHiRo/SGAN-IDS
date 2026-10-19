import pytest

from sgan_ids.config import DataConfig
from sgan_ids.data import prepare


@pytest.fixture(scope="session")
def toy_data():
    return prepare(DataConfig(dataset="toy", test_size=0.2), seed=0)
