"""Tests for the risk manager — the most critical module.

Target: >90% coverage. Test ALL edge cases.
"""

import pytest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

from app.services.risk_manager import Order, RiskManager, RiskVerdict


def make_order(**kwargs) -> Order:
    """Helper to create test orders with sensible defaults."""
    defaults = {
        "symbol": "AAPL",
        "market": "us",
        "side": "buy",
        "order_type": "market",
        "quantity_cents": 5000,  # $50
        "price_cents": 15000,  # $150
        "stop_loss_cents": 14250,  # $142.50 (5% stop)
        "strategy": "momentum",
        "reason": "RSI crossover signal",
    }
    defaults.update(kwargs)
    return Order(**defaults)


class TestRiskManagerBasics:
    def test_approve_valid_order(self):
        rm = RiskManager()
        order = make_order()
        result = rm.evaluate(order)
        assert result.approved is True
        assert result.reason == "All risk checks passed"

    def test_reject_missing_stop_loss(self):
        rm = RiskManager()
        order = make_order(stop_loss_cents=0)
        result = rm.evaluate(order)
        assert result.approved is False
        assert "stop-loss" in result.reason.lower()

    def test_reject_negative_stop_loss(self):
        rm = RiskManager()
        order = make_order(stop_loss_cents=-100)
        result = rm.evaluate(order)
        assert result.approved is False


class TestPositionSizeLimits:
    def test_downsize_oversized_position(self):
        """Oversized orders are cut to the max position size instead of rejected."""
        rm = RiskManager()
        order = make_order(quantity_cents=20000)  # $200 > $100 max
        result = rm.evaluate(order)
        assert result.approved is True
        assert result.adjusted_quantity_cents == 10000  # $100 max position

    def test_approve_max_position(self):
        rm = RiskManager()
        order = make_order(quantity_cents=10000)  # $100 exactly at max
        result = rm.evaluate(order)
        assert result.approved is True

    def test_approve_under_max_position(self):
        rm = RiskManager()
        order = make_order(quantity_cents=5000)  # $50
        result = rm.evaluate(order)
        assert result.approved is True


class TestPerTradeRisk:
    def test_downsize_to_per_trade_risk_limit(self):
        """A position risking more than 2% of the portfolio is shrunk to fit the limit."""
        rm = RiskManager()
        rm.set_portfolio_value(100_000)  # $1,000 -> max risk 2% = 2000 cents
        # 50% stop distance on a $100 position risks 5000 cents > 2000
        order = make_order(quantity_cents=10000, price_cents=15000, stop_loss_cents=7500)
        result = rm.evaluate(order)
        assert result.approved is True
        # 2000 cents of risk / 50% stop distance = 4000 cents
        assert result.adjusted_quantity_cents == 4000

    def test_reject_when_minimum_position_exceeds_per_trade_risk(self):
        """If even a $1 position would breach the per-trade risk limit, reject."""
        rm = RiskManager()
        rm.set_portfolio_value(1_000)  # $10 -> max risk 2% = 20 cents
        # 50% stop: a safe size would be 40 cents, below the $1 minimum
        order = make_order(quantity_cents=10000, price_cents=15000, stop_loss_cents=7500)
        result = rm.evaluate(order)
        assert result.approved is False
        assert "per-trade risk" in result.reason.lower()


class TestCircuitBreaker:
    def test_pause_after_consecutive_losses(self):
        rm = RiskManager()
        rm.record_trade_result(-100)
        rm.record_trade_result(-100)
        rm.record_trade_result(-100)  # 3rd loss triggers pause

        assert rm.is_paused is True

        order = make_order()
        result = rm.evaluate(order)
        assert result.approved is False
        assert "circuit breaker" in result.reason.lower()

    def test_win_resets_consecutive_losses(self):
        rm = RiskManager()
        rm.record_trade_result(-100)
        rm.record_trade_result(-100)
        rm.record_trade_result(500)  # win resets counter
        rm.record_trade_result(-100)  # only 1 loss now

        assert rm.is_paused is False

    def test_pause_expires(self):
        rm = RiskManager()
        rm.record_trade_result(-100)
        rm.record_trade_result(-100)
        rm.record_trade_result(-100)

        # Manually set pause to the past
        rm._paused_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert rm.is_paused is False


class TestKillSwitch:
    def test_kill_switch_halts_trading(self):
        rm = RiskManager()
        rm.kill_switch()

        assert rm.is_halted is True
        order = make_order()
        result = rm.evaluate(order)
        assert result.approved is False
        assert "kill switch" in result.reason.lower()

    def test_reset_after_kill(self):
        rm = RiskManager()
        rm.kill_switch()
        rm.reset_halt()

        assert rm.is_halted is False
        order = make_order()
        result = rm.evaluate(order)
        assert result.approved is True


class TestDailyDrawdown:
    def test_halt_on_daily_drawdown(self):
        rm = RiskManager()
        rm.set_portfolio_value(1_000_000)  # $10,000
        # Max 5% daily loss = $500 = 50000 cents
        rm._daily_pnl_cents = -50000

        order = make_order()
        result = rm.evaluate(order)
        assert result.approved is False
        assert rm.is_halted is True

    def test_reset_daily_pnl(self):
        rm = RiskManager()
        rm._daily_pnl_cents = -30000
        rm.reset_daily_pnl()
        assert rm._daily_pnl_cents == 0
