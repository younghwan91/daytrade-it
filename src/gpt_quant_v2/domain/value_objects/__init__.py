"""Domain Value Objects"""

from gpt_quant_v2.domain.value_objects.base import ValueObject
from gpt_quant_v2.domain.value_objects.confidence import Confidence
from gpt_quant_v2.domain.value_objects.percentage import Percentage
from gpt_quant_v2.domain.value_objects.price import Price
from gpt_quant_v2.domain.value_objects.sentiment import Sentiment, SentimentLabel
from gpt_quant_v2.domain.value_objects.ticker import Ticker
from gpt_quant_v2.domain.value_objects.volume import Volume

__all__ = [
    "ValueObject",
    "Confidence",
    "Percentage",
    "Price",
    "Sentiment",
    "SentimentLabel",
    "Ticker",
    "Volume",
]
