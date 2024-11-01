import unittest
import bittensor as bt
from PIL import Image

from src.protocol import SearchResult
from src.neurons.validator import Validator
from src.neurons.miner import Miner


def get_test_config():
    """Create a config object for testing"""
    config = bt.config()

    # Add neuron config
    config.neuron = bt.config()
    config.neuron.device = "cpu"
    config.neuron.epoch_length = 100
    config.neuron.num_concurrent_forwards = 1
    config.neuron.disable_set_weights = False
    config.neuron.moving_average_alpha = 0.5
    config.neuron.axon_off = True  # Disable axon for testing

    # Add logging config
    config.logging = bt.config()
    config.logging.debug = False
    config.logging.trace = False
    config.logging.record_log = False
    config.logging.logging_dir = 'logs'

    # Add wallet config
    config.wallet = bt.config()
    config.wallet.name = "test_wallet"
    config.wallet.hotkey = "test_hotkey"

    # Mock network for testing
    config.mock = True
    config.netuid = 1

    return config


class TestSearchProtocol(unittest.TestCase):
    """Test the search protocol implementation"""

    def setUp(self):
        # Create test config
        self.config = get_test_config()

        # Create test image
        self.test_image = Image.new('RGB', (100, 100), color='red')

        # Create test search result
        self.test_result = SearchResult.create_from_image(
            image=self.test_image,
            title="Test Title",
            preview="Test Preview",
            url="https://test.com"
        )

    # Rest of test cases remain the same...


class TestMiner(unittest.TestCase):
    """Test the miner implementation"""

    def setUp(self):
        self.config = get_test_config()
        self.miner = Miner(self.config)

    # Rest of test cases remain the same...


class TestValidator(unittest.TestCase):
    """Test the validator implementation"""

    def setUp(self):
        self.config = get_test_config()
        self.validator = Validator(self.config)

    # Rest of test cases remain the same...


if __name__ == '__main__':
    unittest.main()