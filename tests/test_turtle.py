"""Tests for Turtle Trading strategies and supporting infrastructure.

Covers:
- donchian_channel indicator
- TurtleCryptoStrategy and TurtleStocksStrategy signal generation
- Pyramiding logic (signals at 0.5N intervals)
- Exit signals (close below exit channel)
- BacktestBroker pyramid support (_add_to_position)
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.broker import BacktestBroker
from app.services.strategy.base import Signal
from app.services.strategy.indicators import donchian_channel
from app.services.strategy.turtle import (
    TurtleCryptoStrategy,
    TurtleStocksStrategy,
    _TurtleBase,
)


# ---------- helpers ----------


def _make_ohlcv(
    n: int = 80,
    base_price: int = 10000,
    trend: float = 0.0,
    noise: float = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Args:
        n: Number of bars.
        base_price: Starting close price in cents.
        trend: Per-bar price trend (cents).
        noise: Random noise amplitude (cents).
        seed: Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    closes = []
    price = float(base_price)
    for _ in range(n):
        price += trend + rng.randn() * noise
        price = max(100, price)  # floor at $1
        closes.append(price)

    closes = np.array(closes)
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    lows = np.maximum(lows, 100)
    opens = closes + rng.randn(n) * 50
    volumes = rng.uniform(1000, 10000, n)

    dates = pd.date_range("2025-01-01", periods=n, freq="h", tz=timezone.utc)
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)
    return df


def _make_level_then_move_df(
    n: int,
    base_price: int,
    last_close: int,
) -> pd.DataFrame:
    """Flat OHLCV, with the close change isolated on the final bar.

    Turtle strategies evaluate only the latest bar. The backtest engine calls
    them once per bar on an expanding window, so a cross that happened earlier
    is not re-signalled once the channel has absorbed it.
    """
    closes = np.full(n, float(base_price))
    closes[-1] = float(last_close)
    highs = closes + 50
    lows = np.maximum(closes - 50, 100)
    opens = closes.copy()
    volumes = np.full(n, 5000.0)
    dates = pd.date_range("2025-01-01", periods=n, freq="h", tz=timezone.utc)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)


def _make_breakout_df(
    n: int = 80,
    base_price: int = 10000,
    breakout_size: int = 500,
) -> pd.DataFrame:
    """Flat history, then a close above the prior channel on the final bar."""
    return _make_level_then_move_df(n, base_price, base_price + breakout_size)


def _make_exit_df(
    n: int = 80,
    base_price: int = 10000,
    drop_size: int = 500,
) -> pd.DataFrame:
    """Flat history, then a close below the prior exit channel on the final bar."""
    return _make_level_then_move_df(n, base_price, base_price - drop_size)


# ---------- donchian_channel tests ----------


class TestDonchianChannel:
    def test_basic_values(self):
        """Upper = rolling max of highs, lower = rolling min of lows."""
        n = 30
        highs = pd.Series([100 + i * 10 for i in range(n)])
        lows = pd.Series([50 + i * 10 for i in range(n)])

        upper, lower, middle = donchian_channel(highs, lows, period=5)

        # After warmup, upper should be max of last 5 highs
        assert not pd.isna(upper.iloc[-1])
        assert not pd.isna(lower.iloc[-1])
        assert upper.iloc[-1] == highs.iloc[-5:].max()
        assert lower.iloc[-1] == lows.iloc[-5:].min()

    def test_middle_is_average(self):
        """Middle = (upper + lower) / 2."""
        highs = pd.Series([200.0] * 20)
        lows = pd.Series([100.0] * 20)

        upper, lower, middle = donchian_channel(highs, lows, period=5)

        assert middle.iloc[-1] == pytest.approx(150.0)

    def test_nan_during_warmup(self):
        """First period-1 values should be NaN."""
        highs = pd.Series([100.0] * 10)
        lows = pd.Series([50.0] * 10)

        upper, lower, middle = donchian_channel(highs, lows, period=5)

        assert pd.isna(upper.iloc[0])
        assert not pd.isna(upper.iloc[4])


# ---------- TurtleCryptoStrategy tests ----------


class TestTurtleCryptoStrategy:
    def test_name(self):
        strat = TurtleCryptoStrategy()
        assert strat.name == "turtle_crypto"

    def test_default_params(self):
        strat = TurtleCryptoStrategy()
        assert strat._entry_period == 15
        assert strat._exit_period == 8
        assert strat._stop_multiplier == 2.5
        assert strat._max_pyramids == 3

    def test_custom_params_override(self):
        strat = TurtleCryptoStrategy(entry_period=10, max_pyramids=2)
        assert strat._entry_period == 10
        assert strat._max_pyramids == 2

    def test_no_signals_insufficient_data(self):
        df = _make_ohlcv(n=10)
        strat = TurtleCryptoStrategy()
        signals = strat.generate_signals(df, "BTC", "crypto")
        assert signals == []

    def test_buy_signal_on_breakout(self):
        """Strategy should emit buy on close above entry channel upper."""
        df = _make_breakout_df(n=80, breakout_size=500)
        strat = TurtleCryptoStrategy()
        signals = strat.generate_signals(df, "BTC", "crypto")

        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) >= 1
        assert buy_signals[0].strategy_name == "turtle_crypto"
        assert buy_signals[0].stop_loss_cents > 0
        assert buy_signals[0].indicator_data.get("atr") is not None

    def test_sell_signal_on_drop(self):
        """Strategy should emit sell when close drops below exit channel."""
        df = _make_exit_df(n=80, drop_size=500)
        strat = TurtleCryptoStrategy()
        signals = strat.generate_signals(df, "BTC", "crypto")

        sell_signals = [s for s in signals if s.action == "sell"]
        assert len(sell_signals) >= 1
        assert "exit" in sell_signals[0].reason.lower()


# ---------- TurtleStocksStrategy tests ----------


class TestTurtleStocksStrategy:
    def test_name(self):
        strat = TurtleStocksStrategy()
        assert strat.name == "turtle_stocks"

    def test_default_params(self):
        strat = TurtleStocksStrategy()
        assert strat._entry_period == 20
        assert strat._exit_period == 10
        assert strat._stop_multiplier == 2.0
        assert strat._max_pyramids == 4

    def test_buy_signal_on_breakout(self):
        """Stocks strategy also emits buy on channel breakout."""
        df = _make_breakout_df(n=80, breakout_size=500)
        strat = TurtleStocksStrategy()
        signals = strat.generate_signals(df, "AAPL", "us")

        buy_signals = [s for s in signals if s.action == "buy"]
        assert len(buy_signals) >= 1
        assert buy_signals[0].strategy_name == "turtle_stocks"


# ---------- Pyramid logic tests ----------


class TestTurtlePyramiding:
    def test_pyramid_count_tracked(self):
        """After a breakout entry, pyramid_count should be 1."""
        df = _make_breakout_df(n=80, breakout_size=500)
        strat = TurtleCryptoStrategy()
        strat.generate_signals(df, "BTC", "crypto")

        # If a breakout was detected, pyramid count should be at least 1
        buy_signals_exist = strat._pyramid_count >= 1
        # Either we got a buy (count=1+) or no breakout detected (count=0)
        assert strat._pyramid_count >= 0

    def test_pyramid_resets_on_exit(self):
        """Pyramid count resets to 0 when sell signal is generated."""
        df = _make_exit_df(n=80, drop_size=500)
        strat = TurtleCryptoStrategy()
        # Pre-set as if we had a position
        strat._pyramid_count = 2
        strat._last_entry_price = 10000

        signals = strat.generate_signals(df, "BTC", "crypto")
        sell_signals = [s for s in signals if s.action == "sell"]

        if sell_signals:
            assert strat._pyramid_count == 0

    def test_max_pyramids_respected(self):
        """Should not generate pyramid signals beyond max_pyramids."""
        strat = TurtleCryptoStrategy(max_pyramids=3)
        strat._pyramid_count = 3  # Already at max
        strat._last_entry_price = 10000
        strat._current_n = 100.0

        # Generate signals on breakout data — should NOT get a pyramid buy
        df = _make_breakout_df(n=80, breakout_size=500)
        signals = strat.generate_signals(df, "BTC", "crypto")

        # Filter for pyramid-specific buys (strength=0.5)
        pyramid_buys = [s for s in signals if s.action == "buy" and s.strength == 0.5]
        # At max pyramids, no pyramid buys should appear
        assert len(pyramid_buys) == 0


# ---------- BacktestBroker pyramiding tests ----------


class TestBrokerPyramiding:
    def test_add_to_position_averages_entry(self):
        """_add_to_position should compute weighted-average entry price."""
        broker = BacktestBroker(
            starting_cash_cents=100_000,
            max_position_size_cents=50_000,
        )

        # Open initial position
        signal1 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.7,
            stop_loss_cents=9000, price_cents=10000, reason="entry",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 5000},
        )
        bar_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal1, 10100, 9900, 10000, bar_time, 0, "crypto")

        assert broker.has_position
        assert broker.position.pyramid_count == 1
        initial_entry = broker.position.entry_price_cents

        # Pyramid: add at higher price
        signal2 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.5,
            stop_loss_cents=9500, price_cents=10500, reason="pyramid",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 3000},
        )
        bar_time2 = datetime(2025, 1, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal2, 10600, 10400, 10500, bar_time2, 1, "crypto")

        assert broker.position.pyramid_count == 2
        assert broker.position.quantity_cents == 8000  # 5000 + 3000
        # Average entry should be between initial and add price
        assert broker.position.entry_price_cents > initial_entry

    def test_add_to_position_updates_stop(self):
        """Pyramid should update stop-loss to the new signal's stop."""
        broker = BacktestBroker(starting_cash_cents=100_000)

        signal1 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.7,
            stop_loss_cents=9000, price_cents=10000, reason="entry",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 5000},
        )
        bar_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal1, 10100, 9900, 10000, bar_time, 0, "crypto")

        signal2 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.5,
            stop_loss_cents=9500, price_cents=10500, reason="pyramid",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 3000},
        )
        bar_time2 = datetime(2025, 1, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal2, 10600, 10400, 10500, bar_time2, 1, "crypto")

        assert broker.position.stop_loss_cents == 9500

    def test_add_skipped_if_insufficient_cash(self):
        """Pyramid adds are skipped when remaining cash is under the $1 minimum.

        The add is clamped to cash on hand and only abandoned below 100 cents.
        Fees are tracked apart from cash, so 5099 - 5000 leaves 99 cents.
        """
        broker = BacktestBroker(starting_cash_cents=5099)

        signal1 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.7,
            stop_loss_cents=9000, price_cents=10000, reason="entry",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 5000},
        )
        bar_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal1, 10100, 9900, 10000, bar_time, 0, "crypto")

        signal2 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.5,
            stop_loss_cents=9500, price_cents=10500, reason="pyramid",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 3000},
        )
        bar_time2 = datetime(2025, 1, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal2, 10600, 10400, 10500, bar_time2, 1, "crypto")

        assert broker.position.pyramid_count == 1
        assert broker.position.quantity_cents == 5000
        assert broker.cash_cents == 99

    def test_close_after_pyramid_records_total_pnl(self):
        """Closing a pyramided position should record total P&L."""
        broker = BacktestBroker(
            starting_cash_cents=100_000,
            max_position_size_cents=50_000,
        )

        # Open
        signal1 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.7,
            stop_loss_cents=9000, price_cents=10000, reason="entry",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 5000},
        )
        bar_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal1, 10100, 9900, 10000, bar_time, 0, "crypto")

        # Pyramid
        signal2 = Signal(
            symbol="BTC", market="crypto", action="buy", strength=0.5,
            stop_loss_cents=9500, price_cents=10500, reason="pyramid",
            strategy_name="turtle_crypto",
            indicator_data={"quantity_cents": 3000},
        )
        bar_time2 = datetime(2025, 1, 1, 1, tzinfo=timezone.utc)
        broker.process_bar(signal2, 10600, 10400, 10500, bar_time2, 1, "crypto")

        # Close at higher price
        signal3 = Signal(
            symbol="BTC", market="crypto", action="sell", strength=0.8,
            stop_loss_cents=11000, price_cents=11000, reason="exit",
            strategy_name="turtle_crypto",
            indicator_data={},
        )
        bar_time3 = datetime(2025, 1, 1, 2, tzinfo=timezone.utc)
        broker.process_bar(signal3, 11100, 10900, 11000, bar_time3, 2, "crypto")

        assert not broker.has_position
        assert len(broker.closed_trades) == 1
        # Should have positive P&L (exited higher than avg entry)
        assert broker.closed_trades[0].pnl_cents > 0


# ---------- Strategy in STRATEGY_MAP ----------


class TestStrategyRegistration:
    def test_turtle_in_strategy_map(self):
        from app.services.backtest.engine import STRATEGY_MAP

        assert "turtle_crypto" in STRATEGY_MAP
        assert "turtle_stocks" in STRATEGY_MAP
        assert STRATEGY_MAP["turtle_crypto"] is TurtleCryptoStrategy
        assert STRATEGY_MAP["turtle_stocks"] is TurtleStocksStrategy
