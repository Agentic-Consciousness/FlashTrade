"""Tests for Claude AI recommendation feature.

Covers:
- Pydantic model validation (Recommendation, RecommendationSet)
- Prompt building from market context
- Response parsing (valid JSON, markdown-wrapped JSON, malformed)
- API endpoints (GET cache, POST refresh)
- Celery task flow
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.ai.recommender import (
    ClaudeRecommender,
    MarketNews,
    NewsItem,
    Recommendation,
    RecommendationAction,
    RecommendationSet,
)


# ---------- Pydantic model tests ----------


class TestRecommendationModels:
    def test_recommendation_action_values(self):
        assert RecommendationAction.BUY == "buy"
        assert RecommendationAction.SELL == "sell"
        assert RecommendationAction.HOLD == "hold"
        assert RecommendationAction.WATCH == "watch"

    def test_recommendation_valid(self):
        rec = Recommendation(
            symbol="BTC",
            market="crypto",
            action=RecommendationAction.BUY,
            confidence=0.75,
            current_price_cents=15000000,
            entry_price_cents=14800000,
            target_price_cents=16000000,
            stop_loss_cents=14200000,
            reasoning="Strong momentum breakout",
            risk_notes="Resistance at 160K",
            timeframe="1-3 days",
        )
        assert rec.symbol == "BTC"
        assert rec.confidence == 0.75

    def test_recommendation_confidence_bounds(self):
        with pytest.raises(Exception):
            Recommendation(
                symbol="BTC", market="crypto", action="buy",
                confidence=1.5,  # > 1.0, should fail
                current_price_cents=100, reasoning="x", risk_notes="y",
            )

    def test_recommendation_set_defaults(self):
        rec_set = RecommendationSet(
            generated_at_utc="2025-01-01T00:00:00Z",
            model_used="claude-sonnet-4-6",
            market_summary="Markets are mixed.",
            top_opportunities=[],
        )
        assert rec_set.symbols_to_avoid == []
        assert "Not financial advice" in rec_set.disclaimer
        assert rec_set.token_usage == {}
        assert rec_set.crypto_opportunities == []
        assert rec_set.asx_opportunities == []
        assert rec_set.us_opportunities == []
        assert rec_set.uk_opportunities == []
        assert rec_set.market_overview == []

    def test_recommendation_set_with_opportunities(self):
        rec = Recommendation(
            symbol="AAPL", market="us", action="buy", confidence=0.6,
            current_price_cents=23000, reasoning="Oversold bounce",
            risk_notes="Earnings next week",
        )
        rec_set = RecommendationSet(
            generated_at_utc="2025-01-01T00:00:00Z",
            model_used="claude-sonnet-4-6",
            market_summary="Tech recovering.",
            top_opportunities=[rec],
            symbols_to_avoid=["TSLA"],
        )
        assert len(rec_set.top_opportunities) == 1
        assert rec_set.symbols_to_avoid == ["TSLA"]


# ---------- Prompt building tests ----------


class TestPromptBuilding:
    def test_build_user_prompt_with_data(self):
        recommender = ClaudeRecommender()
        context = {
            "timestamp_utc": "2025-06-01T12:00:00Z",
            "symbols": [
                {
                    "symbol": "BTC",
                    "market": "crypto",
                    "price_cents": 15000000,
                    "change_pct": 2.5,
                    "rsi": 45.2,
                    "macd_hist": 5000,
                    "adx": 28.3,
                    "atr_cents": 300000,
                    "bb_position": "within",
                    "regime": "trending",
                    "last_signal": "hold",
                },
                {
                    "symbol": "ETH",
                    "market": "crypto",
                    "data": "insufficient",
                },
            ],
        }
        prompt = recommender._build_user_prompt(context)

        assert "BTC" in prompt
        assert "15000000" in prompt
        assert "insufficient data" in prompt
        assert "3 recs per market (crypto, asx, us, uk)" in prompt

    def test_build_user_prompt_handles_none_indicators(self):
        recommender = ClaudeRecommender()
        context = {
            "timestamp_utc": "2025-06-01T12:00:00Z",
            "symbols": [
                {
                    "symbol": "SOL",
                    "market": "crypto",
                    "price_cents": 500000,
                    "change_pct": -1.0,
                    "rsi": None,
                    "macd_hist": None,
                    "adx": None,
                    "atr_cents": None,
                    "bb_position": "unknown",
                    "regime": "unknown",
                    "last_signal": "hold",
                },
            ],
        }
        prompt = recommender._build_user_prompt(context)
        assert "SOL" in prompt
        assert "--" in prompt  # None values become "--"


# ---------- Response parsing tests ----------


class TestResponseParsing:
    """Test that ClaudeRecommender.generate() correctly parses API responses."""

    SAMPLE_RESPONSE = {
        "market_summary": "Crypto showing strength, stocks mixed.",
        "crypto_opportunities": [
            {
                "symbol": "BTC",
                "market": "crypto",
                "action": "buy",
                "confidence": 0.8,
                "current_price_cents": 15000000,
                "entry_price_cents": 14900000,
                "target_price_cents": 16000000,
                "stop_loss_cents": 14200000,
                "reasoning": "Breakout above 20-day Donchian channel",
                "risk_notes": "Fed meeting this week",
                "timeframe": "2-5 days",
            },
        ],
        "asx_opportunities": [
            {
                "symbol": "BHP.AX",
                "market": "asx",
                "action": "watch",
                "confidence": 0.5,
                "current_price_cents": 4500,
                "reasoning": "Consolidating near support",
                "risk_notes": "Iron ore prices volatile",
                "timeframe": "1-5 days",
            },
        ],
        "us_opportunities": [
            {
                "symbol": "AAPL",
                "market": "us",
                "action": "buy",
                "confidence": 0.65,
                "current_price_cents": 23000,
                "reasoning": "Oversold bounce setup",
                "risk_notes": "Earnings next week",
                "timeframe": "2-5 days",
            },
        ],
        "uk_opportunities": [
            {
                "symbol": "SHEL.L",
                "market": "uk",
                "action": "watch",
                "confidence": 0.55,
                "current_price_cents": 280000,
                "reasoning": "Oil majors under pressure",
                "risk_notes": "Energy sector rotation",
                "timeframe": "1-5 days",
            },
        ],
        "symbols_to_avoid": ["DOGE"],
    }

    @pytest.mark.asyncio
    async def test_parse_clean_json(self):
        """Response is clean JSON (no markdown wrapping)."""
        recommender = ClaudeRecommender()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(self.SAMPLE_RESPONSE))]
        mock_response.usage.input_tokens = 1500
        mock_response.usage.output_tokens = 500

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        recommender._client = mock_client

        with patch.object(recommender, "_gather_context", new_callable=AsyncMock) as mock_ctx:
            mock_ctx.return_value = {
                "timestamp_utc": "2025-01-01T00:00:00Z",
                "symbols": [],
                "market_overview": [],
            }
            result = await recommender.generate()

        assert isinstance(result, RecommendationSet)
        # Per-market arrays
        assert len(result.crypto_opportunities) == 1
        assert result.crypto_opportunities[0].symbol == "BTC"
        assert result.crypto_opportunities[0].confidence == 0.8
        assert len(result.asx_opportunities) == 1
        assert result.asx_opportunities[0].symbol == "BHP.AX"
        assert len(result.us_opportunities) == 1
        assert result.us_opportunities[0].symbol == "AAPL"
        assert len(result.uk_opportunities) == 1
        assert result.uk_opportunities[0].symbol == "SHEL.L"
        # Combined top_opportunities
        assert len(result.top_opportunities) == 4
        assert result.symbols_to_avoid == ["DOGE"]
        assert result.token_usage["input_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_parse_markdown_wrapped_json(self):
        """Response has ```json ... ``` wrapping."""
        recommender = ClaudeRecommender()

        wrapped = "Here's my analysis:\n```json\n" + json.dumps(self.SAMPLE_RESPONSE) + "\n```"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=wrapped)]
        mock_response.usage.input_tokens = 1500
        mock_response.usage.output_tokens = 600

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        recommender._client = mock_client

        with patch.object(recommender, "_gather_context", new_callable=AsyncMock) as mock_ctx:
            mock_ctx.return_value = {"timestamp_utc": "2025-01-01T00:00:00Z", "symbols": [], "market_overview": []}
            result = await recommender.generate()

        assert len(result.top_opportunities) == 4
        assert len(result.crypto_opportunities) == 1
        assert len(result.uk_opportunities) == 1

    @pytest.mark.asyncio
    async def test_parse_invalid_json_raises(self):
        """Malformed JSON should raise ValueError."""
        recommender = ClaudeRecommender()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not JSON at all")]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 50

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        recommender._client = mock_client

        with patch.object(recommender, "_gather_context", new_callable=AsyncMock) as mock_ctx:
            mock_ctx.return_value = {"timestamp_utc": "2025-01-01T00:00:00Z", "symbols": [], "market_overview": []}
            with pytest.raises(ValueError, match="Invalid JSON"):
                await recommender.generate()


# ---------- API key validation ----------


class TestApiKeyValidation:
    def test_no_api_key_raises(self):
        recommender = ClaudeRecommender()
        with patch("app.services.ai.recommender.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not configured"):
                recommender._get_client()


# ---------- RecommendationSet serialization ----------


class TestSerialization:
    def test_model_dump_json(self):
        rec = Recommendation(
            symbol="ETH", market="crypto", action="watch", confidence=0.4,
            current_price_cents=500000, reasoning="Consolidating", risk_notes="Low volume",
        )
        rec_set = RecommendationSet(
            generated_at_utc="2025-01-01T00:00:00Z",
            model_used="claude-sonnet-4-6",
            market_summary="Quiet markets.",
            top_opportunities=[rec],
            crypto_opportunities=[rec],
        )

        json_str = rec_set.model_dump_json()
        data = json.loads(json_str)

        assert data["model_used"] == "claude-sonnet-4-6"
        assert len(data["top_opportunities"]) == 1
        assert len(data["crypto_opportunities"]) == 1
        assert data["top_opportunities"][0]["action"] == "watch"
        assert "Not financial advice" in data["disclaimer"]
        assert data["market_overview"] == []


class TestMarketNewsModels:
    def test_news_item(self):
        item = NewsItem(headline="Markets rise", summary="Stocks advanced broadly.")
        assert item.headline == "Markets rise"
        assert item.summary == "Stocks advanced broadly."

    def test_market_news(self):
        news = MarketNews(
            us_news=NewsItem(headline="US up", summary="S&P gained 1%."),
            global_news=NewsItem(headline="Asia mixed", summary="Nikkei flat."),
            australian_news=NewsItem(headline="ASX flat", summary="Banks led losses."),
            notable_news=NewsItem(headline="Oil spikes", summary="Brent crude +3%."),
            generated_at_utc="2025-01-01T00:00:00Z",
        )
        assert news.us_news.headline == "US up"
        assert news.model_used == "claude-haiku-4-5-20251001"
        assert news.token_usage == {}

    def test_market_news_with_uk_recommendation(self):
        rec = Recommendation(
            symbol="SHEL.L", market="uk", action="buy", confidence=0.7,
            current_price_cents=280000, reasoning="Breakout", risk_notes="Oil vol",
        )
        rec_set = RecommendationSet(
            generated_at_utc="2025-01-01T00:00:00Z",
            model_used="claude-sonnet-4-6",
            market_summary="UK markets strong.",
            top_opportunities=[rec],
            uk_opportunities=[rec],
        )
        assert len(rec_set.uk_opportunities) == 1
        assert rec_set.uk_opportunities[0].symbol == "SHEL.L"
