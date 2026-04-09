"""Tests for ML layer: regime classifier and strategy selector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.models import MarketRegime
from trading_bot.features.regime import build_regime_features, heuristic_regime
from trading_bot.ml.regime_classifier import RegimeClassifier
from trading_bot.ml.strategy_selector import StrategySelector


class TestHeuristicRegime:
    def test_trending_regime(self):
        features = {"adx": 35, "atr_pct": 0.015, "roc_abs": 2.5, "volume_ratio": 1.2}
        assert heuristic_regime(features) == MarketRegime.TRENDING

    def test_breakout_regime(self):
        features = {"adx": 20, "atr_pct": 0.02, "bb_width": 0.10, "volume_ratio": 2.5, "roc_abs": 3.0}
        assert heuristic_regime(features) == MarketRegime.BREAKOUT

    def test_ranging_is_default(self):
        features = {"adx": 15, "atr_pct": 0.005, "roc_abs": 0.5, "volume_ratio": 1.0}
        assert heuristic_regime(features) == MarketRegime.RANGING

    def test_volatile_regime(self):
        features = {"adx": 20, "atr_pct": 0.03, "roc_abs": 1.0, "volume_ratio": 1.2}
        assert heuristic_regime(features) == MarketRegime.VOLATILE


class TestRegimeClassifier:
    def test_untrained_falls_back_to_heuristic(self):
        clf = RegimeClassifier(model_path="/tmp/nonexistent_model.pkl")
        features = {"adx": 35, "roc_abs": 2.0, "atr_pct": 0.01}
        regime, confidence = clf.predict(features)
        assert isinstance(regime, MarketRegime)
        assert 0 < confidence <= 1.0

    def test_train_and_predict(self, tmp_path):
        clf = RegimeClassifier(model_path=str(tmp_path / "clf.pkl"))
        np.random.seed(0)
        X = pd.DataFrame(
            np.random.rand(600, 5),
            columns=["adx", "atr_pct", "rsi", "roc_abs", "volume_ratio"],
        )
        y = pd.Series(np.random.randint(0, 4, 600))
        clf.fit(X, y, save=True)
        assert clf.is_trained
        regime, conf = clf.predict({"adx": 0.5, "atr_pct": 0.3, "rsi": 0.6, "roc_abs": 0.2, "volume_ratio": 0.8})
        assert isinstance(regime, MarketRegime)
        assert 0 < conf <= 1.0

    def test_model_saves_and_loads(self, tmp_path):
        path = str(tmp_path / "clf.pkl")
        clf1 = RegimeClassifier(model_path=path)
        np.random.seed(1)
        X = pd.DataFrame(np.random.rand(600, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randint(0, 4, 600))
        clf1.fit(X, y, save=True)

        clf2 = RegimeClassifier(model_path=path)
        assert clf2.is_trained


class TestStrategySelector:
    def test_weights_sum_to_one(self):
        selector = StrategySelector()
        for regime in MarketRegime:
            weights = selector.get_weights(regime)
            assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_trending_regime_favors_trend(self):
        selector = StrategySelector()
        weights = selector.get_weights(MarketRegime.TRENDING)
        assert weights.get("trend_following", 0) == max(weights.values())

    def test_ranging_regime_favors_mean_reversion(self):
        selector = StrategySelector()
        weights = selector.get_weights(MarketRegime.RANGING)
        assert weights.get("mean_reversion", 0) == max(weights.values())

    def test_breakout_regime_favors_breakout(self):
        selector = StrategySelector()
        weights = selector.get_weights(MarketRegime.BREAKOUT)
        assert weights.get("breakout", 0) == max(weights.values())
