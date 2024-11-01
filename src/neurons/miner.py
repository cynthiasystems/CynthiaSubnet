import time
import typing
import bittensor as bt
from PIL import Image
import aiohttp
import asyncio

from src.neurons.base.miner import BaseMinerNeuron
from src.protocol import SearchSynapse, SearchResult


class Miner(BaseMinerNeuron):
    """
    Cynthia search miner that processes search queries and returns relevant results
    """

    def __init__(self, config=None):
        super().__init__(config=config)

        # Initialize any search indexes or APIs here
        self.session = aiohttp.ClientSession()

    async def forward(
            self, synapse: SearchSynapse
    ) -> SearchSynapse:
        """
        Processes the search query and returns relevant results

        Args:
            synapse: SearchSynapse containing the query

        Returns:
            synapse with filled result field containing SearchResult
        """
        try:
            # TODO: Implement actual search logic here
            # This is a mock implementation

            # Mock getting an image from URL
            async with self.session.get("https://example.com/image.jpg") as response:
                if response.status == 200:
                    img_data = await response.read()
                    image = Image.open(io.BytesIO(img_data))

                    # Create search result
                    synapse.result = SearchResult.create_from_image(
                        image=image,
                        title="Example Search Result",
                        preview="This is a preview of the search result...",
                        url="https://example.com"
                    )

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            synapse.result = None

        return synapse

    async def blacklist(
            self, synapse: SearchSynapse
    ) -> typing.Tuple[bool, str]:
        """Determines whether an incoming request should be blacklisted"""

        if not synapse.query or len(synapse.query.strip()) == 0:
            return True, "Empty query"

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unauthorized hotkey"

        return False, "Authorized"

    async def priority(self, synapse: SearchSynapse) -> float:
        """Assigns priority score based on caller's stake"""
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        return priority


# Main loop
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
