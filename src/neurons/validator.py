import time
import random
import bittensor as bt
import numpy as np
from typing import List

from src.neurons.base.validator import BaseValidatorNeuron
from src.protocol import SearchSynapse, SearchResult


class Validator(BaseValidatorNeuron):
    """
    Cynthia search validator that queries miners and scores their results
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.load_state()

    async def forward(self):
        """
        The validator's forward pass, queries miners and scores responses
        """

        # Get random uids from metagraph (limit to 10 for testing)
        uids = random.sample(range(self.metagraph.n), min(10, self.metagraph.n))

        # Create search synapse with a test query
        synapse = SearchSynapse(
            query="test search query"
        )

        # Query the network
        responses = await self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in uids],
            synapse=synapse,
            deserialize=True,
            timeout=10
        )

        # Score the responses
        rewards = self.score_responses(responses)
        bt.logging.info(f"Scores: {rewards}")

        # Update the scores for each miner
        self.update_scores(rewards, uids)

    def score_responses(self, responses: List[SearchResult]) -> np.array:
        """Score miner responses based on result quality"""

        scores = []
        for response in responses:
            try:
                # No result returned
                if response is None:
                    scores.append(0.0)
                    continue

                score = 0.0

                # Check if image was returned and valid
                if response.image_base64:
                    score += 0.5

                # Check if title and preview are non-empty
                if response.title and response.preview_text:
                    score += 0.3

                # Check if URL is valid
                if response.host_url:
                    score += 0.2

                scores.append(score)

            except Exception as e:
                bt.logging.error(f"Error scoring response: {e}")
                scores.append(0.0)

        return np.array(scores)


# Main loop
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
