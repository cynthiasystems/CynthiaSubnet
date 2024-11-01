from typing import Optional
from dataclasses import dataclass
import base64
import bittensor as bt
from PIL import Image
import io


@dataclass
class SearchResult:
    """Represents a single search result with image, title, preview and metadata"""
    title: str
    preview_text: str
    image_base64: str  # Base64 encoded image
    host_url: str

    @staticmethod
    def create_from_image(image: Image.Image, title: str, preview: str, url: str) -> "SearchResult":
        """Creates a SearchResult from a PIL Image and metadata"""
        # Convert PIL image to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode()

        return SearchResult(
            title=title,
            preview_text=preview,
            image_base64=img_base64,
            host_url=url
        )

    def to_image(self) -> Optional[Image.Image]:
        """Converts the base64 image back to PIL Image"""
        try:
            img_data = base64.b64decode(self.image_base64)
            return Image.open(io.BytesIO(img_data))
        except:
            return None


class SearchSynapse(bt.Synapse):
    """Handles search queries and results between miners and validators"""

    # Required request fields
    query: str  # The search query text

    # Optional response fields filled by miner
    result: Optional[SearchResult] = None

    def deserialize(self) -> Optional[SearchResult]:
        """Deserialize the response result"""
        return self.result
