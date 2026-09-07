"""Prompt templates for OpenAI sentiment analysis."""

SENTIMENT_SYSTEM_PROMPT = """\
You are a financial news sentiment analyst specializing in stock market analysis.
Your task is to analyze news articles and determine their sentiment impact on the mentioned stocks.

Guidelines:
1. Focus on actionable market signals, not general news tone
2. Consider both direct and indirect market implications
3. Weight recent events more heavily than historical context
4. Identify specific catalysts that could move stock prices
5. Be objective and avoid bias from prior market sentiment

Output your analysis in the specified JSON format."""

SENTIMENT_USER_PROMPT_TEMPLATE = """\
Analyze the following financial news article for market sentiment.

**Title:** {title}

**Source:** {source}

**Published:** {published_at}

**Content:**
{content}

{tickers_section}

Provide your analysis in JSON format with the following structure:
{{
    "sentiment_score": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "sentiment_label": <"BULLISH" | "BEARISH" | "NEUTRAL">,
    "confidence": <float from 0.0 to 1.0 indicating how confident you are>,
    "key_points": [<list of 2-5 key points that influenced your analysis>],
    "reasoning": <brief explanation of your sentiment determination>
}}

Scoring Guidelines:
- BULLISH (score > 0.2): Positive catalysts, growth drivers, favorable news
- BEARISH (score < -0.2): Negative catalysts, risks, unfavorable developments
- NEUTRAL (-0.2 <= score <= 0.2): Mixed signals, routine news, no clear direction

Be precise and concise in your analysis."""


def format_sentiment_user_prompt(
    title: str,
    content: str,
    source: str,
    published_at: str,
    tickers: list[str] | None = None,
) -> str:
    """Format the sentiment analysis user prompt with article details.

    Args:
        title: Article headline
        content: Full article text
        source: News source/publisher
        published_at: Publication date string
        tickers: Optional list of mentioned stock tickers

    Returns:
        Formatted prompt string
    """
    tickers_section = ""
    if tickers:
        tickers_section = f"**Mentioned Tickers:** {', '.join(tickers)}\n"

    return SENTIMENT_USER_PROMPT_TEMPLATE.format(
        title=title,
        content=content,
        source=source,
        published_at=published_at,
        tickers_section=tickers_section,
    )


# JSON schema for structured output (function calling)
SENTIMENT_ANALYSIS_SCHEMA = {
    "name": "sentiment_analysis",
    "description": "Analyze financial news sentiment for stock market impact",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment_score": {
                "type": "number",
                "description": "Sentiment score from -1.0 (very bearish) to 1.0 (very bullish)",
            },
            "sentiment_label": {
                "type": "string",
                "enum": ["BULLISH", "BEARISH", "NEUTRAL"],
                "description": "Categorical sentiment label",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level from 0.0 to 1.0",
            },
            "key_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key points that influenced the sentiment analysis",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the sentiment determination",
            },
        },
        "required": ["sentiment_score", "sentiment_label", "confidence", "key_points", "reasoning"],
        "additionalProperties": False,
    },
}


__all__ = [
    "SENTIMENT_SYSTEM_PROMPT",
    "SENTIMENT_USER_PROMPT_TEMPLATE",
    "SENTIMENT_ANALYSIS_SCHEMA",
    "format_sentiment_user_prompt",
]
