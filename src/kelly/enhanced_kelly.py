"""
增强 Kelly 仓位引擎

融合 GBDT 胜率 + Kronos 趋势/波动率 → 动态 Kelly 仓位计算。
保留现有系统的所有安全机制 (Half-Kelly, cap, min/max)。
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    KELLY_DEFAULT_WIN_RATE, KELLY_MAX_WIN_RATE_CAP,
    KELLY_MIN_POSITION_SOL, KELLY_MAX_POSITION_PCT,
    KELLY_ODDS_CAP, KELLY_FRACTION,
)


class EnhancedKelly:
    """
    增强版 Kelly Criterion 仓位计算器。

    相比现有系统的 entry_engine.py 的改进:
    1. 胜率 p 来自 GBDT 模型 (数据驱动), 而非手工规则
    2. 赔率 b 来自 Kronos 预测的 upside/downside, 而非历史平均
    3. 保留 Half-Kelly 和所有安全上限
    """

    def __init__(self, total_capital_sol: float = 5.0):
        self.total_capital = total_capital_sol

    def calculate(self, gbdt_output: dict, kronos_output: dict,
                  historical_odds: float = None) -> dict:
        """
        计算 Kelly 仓位。

        Args:
            gbdt_output: GBDT 预测结果
                {win_probability, expected_pnl, confidence}
            kronos_output: Kronos 预测结果
                {trend_direction, upside, downside, confidence, implied_volatility}
            historical_odds: 历史赔率 (可选, 用于混合)

        Returns:
            {
                "position_size_sol": float,
                "kelly_fraction": float,
                "win_probability": float,
                "odds_ratio": float,
                "decision": str,  # ENTER / SKIP / WAIT
                "reasoning": str,
            }
        """
        # ============================================================
        # Step 1: 估算胜率 p (GBDT 为主, Kronos 趋势为辅)
        # ============================================================
        p_gbdt = gbdt_output.get("win_probability", KELLY_DEFAULT_WIN_RATE)

        # Kronos 趋势方向作为修正因子
        kronos_trend = kronos_output.get("trend_direction", 0.0)
        kronos_conf = kronos_output.get("confidence", 0.0)

        # 如果 Kronos 预测看涨 + 高置信度, 微调胜率
        if kronos_trend > 0.01 and kronos_conf > 0.5:
            p_adjustment = min(0.05, kronos_trend * kronos_conf * 0.1)
            p = p_gbdt + p_adjustment
        elif kronos_trend < -0.01 and kronos_conf > 0.5:
            p_adjustment = max(-0.05, kronos_trend * kronos_conf * 0.1)
            p = p_gbdt + p_adjustment
        else:
            p = p_gbdt

        # 安全上限
        p = min(p, KELLY_MAX_WIN_RATE_CAP)
        p = max(p, 0.05)  # 最低 5%
        q = 1 - p

        # ============================================================
        # Step 2: 估算赔率 b (Kronos 为主, 历史为辅)
        # ============================================================
        upside = kronos_output.get("upside", 0.0)
        downside = kronos_output.get("downside", 0.0)

        if downside > 0.01 and upside > 0:
            b_kronos = upside / downside
        elif historical_odds and historical_odds > 0:
            b_kronos = historical_odds
        else:
            b_kronos = 1.5  # 默认假设 1.5:1

        # 如果有历史赔率, 混合使用 (70% Kronos + 30% 历史)
        if historical_odds and historical_odds > 0:
            b = 0.7 * b_kronos + 0.3 * historical_odds
        else:
            b = b_kronos

        # 安全上限
        b = min(b, KELLY_ODDS_CAP)
        b = max(b, 0.1)

        # ============================================================
        # Step 3: Kelly 公式
        # ============================================================
        kelly_raw = (p * b - q) / b

        # 负 Kelly = 不应该入场
        if kelly_raw <= 0:
            return {
                "position_size_sol": 0.0,
                "kelly_fraction": 0.0,
                "win_probability": p,
                "odds_ratio": b,
                "decision": "SKIP",
                "reasoning": f"Kelly 为负 ({kelly_raw:.4f}): p={p:.3f}, b={b:.2f} → EV为负, 不入场",
            }

        # Half-Kelly
        kelly = kelly_raw * KELLY_FRACTION

        # 仓位计算
        position_sol = kelly * self.total_capital
        position_sol = max(position_sol, KELLY_MIN_POSITION_SOL)
        position_sol = min(position_sol, self.total_capital * KELLY_MAX_POSITION_PCT)

        # ============================================================
        # Step 4: 决策
        # ============================================================
        gbdt_confidence = gbdt_output.get("confidence", "LOW")
        implied_vol = kronos_output.get("implied_volatility", 0.05)

        # 高波动率时额外缩减仓位 (波动率惩罚)
        if implied_vol > 0.10:
            vol_penalty = max(0.5, 1.0 - (implied_vol - 0.10) * 2)
            position_sol *= vol_penalty

        # 决策逻辑
        if gbdt_confidence == "LOW" and kronos_trend < 0:
            decision = "SKIP"
            reasoning = f"GBDT 低信心 + Kronos 看跌 → 跳过"
        elif gbdt_confidence == "FALLBACK":
            decision = "WAIT"
            reasoning = f"GBDT 模型不可用, 降级到默认参数"
        else:
            decision = "ENTER"
            reasoning = (
                f"p={p:.3f}(GBDT={p_gbdt:.3f}), b={b:.2f}, "
                f"Kelly={kelly_raw:.4f}→{kelly:.4f}, "
                f"仓位={position_sol:.4f}SOL, "
                f"Kronos趋势={kronos_trend:+.3f}(conf={kronos_conf:.2f})"
            )

        return {
            "position_size_sol": round(position_sol, 4),
            "kelly_fraction": round(kelly, 6),
            "kelly_raw": round(kelly_raw, 6),
            "win_probability": round(p, 4),
            "odds_ratio": round(b, 4),
            "decision": decision,
            "reasoning": reasoning,
            "components": {
                "p_gbdt": round(p_gbdt, 4),
                "p_final": round(p, 4),
                "b_kronos": round(b_kronos, 4),
                "b_historical": historical_odds,
                "b_final": round(b, 4),
                "implied_volatility": round(implied_vol, 6),
                "kronos_trend": round(kronos_trend, 4),
                "kronos_confidence": round(kronos_conf, 4),
            }
        }
