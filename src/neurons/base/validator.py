from abc import ABC
import asyncio
import copy
import bittensor as bt
from typing import List
import numpy as np

from src.neurons.base.neuron import BaseNeuron


class BaseValidatorNeuron(BaseNeuron, ABC):
    """Base class for validators."""

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save hotkeys copy
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Setup dendrite
        if self.config.mock:
            self.dendrite = bt.MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)

        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Initialize scores as numpy array
        self.scores = np.zeros(self.metagraph.n)

        # Sync with network
        self.sync()

        # Serve axon if enabled
        if not self.config.neuron.axon_off:
            self.serve_axon()

        # Setup async loop
        self.loop = asyncio.get_event_loop()

        # Setup background thread
        self.should_exit = False
        self.is_running = False
        self.thread = None
        self.lock = asyncio.Lock()

    def serve_axon(self):
        """Serve axon to network."""
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(f"Serving validator axon {self.axon}")
            except Exception as e:
                bt.logging.error(f"Failed to serve axon: {e}")

        except Exception as e:
            bt.logging.error(f"Failed to create axon: {e}")

    def run(self):
        """Main validator loop."""
        try:
            while not self.should_exit:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run concurrent forwards
                self.loop.run_until_complete(self.concurrent_forward())

                # Sync and update weights
                self.sync()

                self.step += 1

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        except Exception as err:
            bt.logging.error(f"Error in validator loop: {str(err)}")

    async def concurrent_forward(self):
        """Run multiple forward passes concurrently."""
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def update_scores(self, rewards: np.ndarray, uids: List[int]):
        """Update scores using exponential moving average."""
        # Handle NaN values
        if np.isnan(rewards).any():
            bt.logging.warning("NaN values in rewards")
            rewards = np.nan_to_num(rewards, 0.0)

        # Ensure numpy arrays
        rewards = np.asarray(rewards)
        uids = np.asarray(uids)

        # Handle empty arrays
        if rewards.size == 0 or uids.size == 0:
            bt.logging.warning("Empty rewards or uids")
            return

        # Update scores with moving average
        scattered_rewards = np.zeros_like(self.scores)
        scattered_rewards[uids] = rewards

        alpha = self.config.neuron.moving_average_alpha
        self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores

    def set_weights(self):
        """Set weights on chain."""
        weights = np.copy(self.scores)
        weights = weights / np.sum(weights)  # Normalize

        # Convert to uint16
        max_value = 65535
        uint_weights = (weights * max_value).astype('uint16')

        # Set on chain
        try:
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                weights=uint_weights,
                uids=self.metagraph.uids,
                wait_for_inclusion=False
            )
            bt.logging.info("Successfully set weights")
        except Exception as e:
            bt.logging.error(f"Failed to set weights: {e}")
