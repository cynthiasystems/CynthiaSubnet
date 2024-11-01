import asyncio
import bittensor as bt

from src.neurons.base.neuron import BaseNeuron


class BaseMinerNeuron(BaseNeuron):
    """Base class for miners."""

    def __init__(self, config=None):
        super().__init__(config=config)

        # Setup axon
        if self.config.mock:
            self.axon = bt.MockAxon(wallet=self.wallet)
        else:
            self.axon = bt.axon(wallet=self.wallet)

        # Attach handlers
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority
        )

        # Start serving
        if not self.config.neuron.axon_off:
            self.axon.start()
            self.subtensor.serve_axon(
                netuid=self.config.netuid,
                axon=self.axon
            )

        # Setup background thread
        self.should_exit = False
        self.is_running = False
        self.thread = None

    def run(self):
        """Main miner loop."""
        try:
            while not self.should_exit:
                # Sync with network
                self.sync()

                # Sleep briefly
                asyncio.sleep(1)

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        except Exception as err:
            bt.logging.error(f"Error in miner loop: {str(err)}")
            self.axon.stop()

    async def blacklist(self, synapse) -> bool:
        """Default blacklist implementation."""
        return False

    async def priority(self, synapse) -> float:
        """Default priority implementation."""
        return 1.0
