import copy
import typing
import bittensor as bt
from abc import ABC, abstractmethod


def ttl_get_block(self):
    """Gets current block from cache or chain."""
    if hasattr(self, '_block'):
        return self._block
    self._block = self.subtensor.block
    return self._block


class BaseNeuron(ABC):
    """Base class for Bittensor neurons."""

    def __init__(self, config=None):
        # Build default config if none provided
        if config is None:
            config = bt.config()
            config.neuron.device = "cpu"
            config.neuron.epoch_length = 100
            config.neuron.num_concurrent_forwards = 1
            config.neuron.disable_set_weights = False
            config.neuron.moving_average_alpha = 0.5
            config.neuron.axon_off = False

            # Add logging config
            config.logging = bt.config()
            config.logging.debug = False
            config.logging.trace = False
            config.logging.record_log = False
            config.logging.logging_dir = 'logs'

        self.config = config

        # Set up logging with the provided configuration.
        bt.logging.set_config(config=self.config.logging)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        if self.config.mock:
            self.wallet = bt.MockWallet(config=self.config)
            self.subtensor = bt.MockSubtensor(self.config.netuid)
            self.metagraph = bt.MockMetagraph(self.config.netuid)
        else:
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check registration
        self.check_registered()

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running neuron on subnet: {self.config.netuid} with uid {self.uid}")
        self.step = 0

    @property
    def block(self):
        """Get current block."""
        return ttl_get_block(self)

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        ...

    @abstractmethod
    def run(self):
        ...

    def check_registered(self):
        """Check if wallet is registered."""
        if not self.subtensor.is_hotkey_registered(
                netuid=self.config.netuid,
                hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}")
            exit()

    def sync(self):
        """Synchronize neuron state with network."""
        self.check_registered()

        # Check for updates
        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()

        self.save_state()

    def should_sync_metagraph(self):
        """Check if metagraph should be synced."""
        return (self.block - self.metagraph.last_update[self.uid]) > self.config.neuron.epoch_length

    def should_set_weights(self):
        """Check if weights should be set."""
        # Skip first step
        if self.step == 0:
            return False

        # Check if disabled
        if self.config.neuron.disable_set_weights:
            return False

        # Check epoch length
        return (self.block - self.metagraph.last_update[self.uid]) > self.config.neuron.epoch_length

    def save_state(self):
        """Save neuron state."""
        pass

    def load_state(self):
        """Load neuron state."""
        pass