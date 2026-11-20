import pytest

from decision_helpers import toy_products, toy_redemptions, toy_table
from decision.menu import Lab


@pytest.fixture
def lab() -> Lab:
    return Lab(toy_table(), toy_products(), toy_redemptions())
