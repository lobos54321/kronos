"""
K 线趋势预测引擎

两种模式:
1. 统计模式 (默认): 从 K 线数据直接计算趋势/波动率, 零依赖
2. Kronos 模式: 用 Transformer 基础模型预测 (需要 PyTorch + 2GB RAM)

当前默认使用统计模式, 因为:
- Zeabur 免费方案内存不够跑 PyTorch
- 对 meme coin 这种高噪声资产, 统计方法 vs Transformer 差距可能不大
- 等验证有 alpha 后再升级到 Kronos
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import KLINE_DB_PATH, DATA_MODE, KRONOS_MAX_CONTEXT


class KronosPredictor:
    """
    K 线趋势预测器。

    输出 (与 Kronos Transformer 接口完全一致):
    - trend_direction: >0 看涨, <0 看跌
    - trend_magnitude: 趋势幅度 (%)
    - implied_volatility: 隐含波动率
    - confidence: 预测置信度 (0-1)
    - upside / downside: 预期上行/下行幅度
    """

    def __init__(self, device: str = None, api_client=None):
        self._api_client = api_client

    def load(self) -> bool:
        """统计模式不需要加载模型"""
        return True

    def predict_for_token(self, token_ca: str) -> dict:
        """
        获取指定 token 的 K 线并计算趋势预测。
        """
        kline_df = self._get_klines(token_ca)

        if kline_df is None or len(kline_df) < 5:
            return self._fallback_prediction()

        return self._statistical_predict(kline_df)

    def _statistical_predict(self, df: pd.DataFrame) -> dict:
        """
        从历史 K 线统计计算趋势预测。
        
        替代 Kronos Transformer, 输出格式完全一致。
        """
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        n = len(closes)

        current_price = closes[-1]

        # ── 趋势方向 (线性回归斜率) ──
        x = np.arange(n)
        if n >= 5 and np.std(closes) > 0:
            slope = np.polyfit(x, closes, 1)[0]
            # 用最近 1/3 数据的斜率 (更敏感)
            recent_n = max(5, n // 3)
            recent_slope = np.polyfit(x[-recent_n:], closes[-recent_n:], 1)[0]
            # 综合: 70% 近期 + 30% 整体
            avg_slope = 0.7 * recent_slope + 0.3 * slope
            trend_direction = float(avg_slope * n / (current_price + 1e-15))
        else:
            trend_direction = 0.0

        trend_magnitude = abs(trend_direction) * 100

        # ── 隐含波动率 (high-low range / close) ──
        if n >= 3:
            ranges = highs - lows
            implied_vol = float(np.mean(ranges / (closes + 1e-15)))
        else:
            implied_vol = 0.05

        # ── 上行/下行估计 ──
        if n >= 10:
            # 用最近 N 根 K 线的 rolling max/min
            recent = min(60, n)
            recent_highs = highs[-recent:]
            recent_lows = lows[-recent:]
            upside = float((np.max(recent_highs) - current_price) / (current_price + 1e-15))
            downside = float((current_price - np.min(recent_lows)) / (current_price + 1e-15))
        else:
            upside = implied_vol
            downside = implied_vol

        # ── 置信度 (趋势一致性) ──
        if n >= 5:
            # 阳线占比 (最近 20 根)
            recent_n = min(20, n)
            recent_opens = opens[-recent_n:]
            recent_closes = closes[-recent_n:]
            green_ratio = np.mean(recent_closes >= recent_opens)

            # 趋势一致性: 如果趋势向上, 阳线越多越有信心
            if trend_direction > 0:
                confidence = float(green_ratio)
            else:
                confidence = float(1.0 - green_ratio)
        else:
            confidence = 0.3

        return {
            "trend_direction": trend_direction,
            "trend_magnitude": trend_magnitude,
            "implied_volatility": max(0.001, implied_vol),
            "confidence": min(1.0, max(0.0, confidence)),
            "pred_df": None,  # 统计方法不生成预测 DataFrame
            "upside": max(0, upside),
            "downside": max(0, downside),
        }

    def _get_klines(self, token_ca: str) -> pd.DataFrame:
        """
        多源获取 K 线。
        优先 kline_cache.db, 不够则从 GeckoTerminal 补充。
        """
        df = None

        # 策略 1: 从现有系统的 kline_cache.db 读取
        if DATA_MODE == "api" and self._api_client:
            try:
                db_path = self._api_client.download_kline_db()
                if db_path:
                    df = self._read_klines_from_db(token_ca, db_path)
            except Exception as e:
                print(f"   ⚠️ 下载 K 线缓存失败: {e}")
        else:
            df = self._read_klines_from_db(token_ca, KLINE_DB_PATH)

        if df is not None and len(df) >= 10:
            return df

        # 策略 2: 从 GeckoTerminal 获取
        return self._fetch_from_gecko(token_ca)

    def _fetch_from_gecko(self, token_ca: str) -> pd.DataFrame:
        """从 GeckoTerminal 免费 API 获取 K 线"""
        import time
        import requests

        headers = {"Accept": "application/json", "User-Agent": "KronosShadow/1.0"}

        try:
            r = requests.get(
                f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_ca}/pools",
                headers=headers, timeout=15,
                params={"page": "1", "sort": "h24_volume_usd_desc"}
            )
            if r.status_code == 429:
                time.sleep(30)
                r = requests.get(
                    f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_ca}/pools",
                    headers=headers, timeout=15,
                    params={"page": "1", "sort": "h24_volume_usd_desc"}
                )
            if r.status_code != 200:
                return None

            pools = r.json().get("data", [])
            if not pools:
                return None

            pool_id = pools[0].get("id", "").replace("solana_", "")
            pool_name = pools[0].get("attributes", {}).get("name", "")
        except Exception:
            return None

        time.sleep(2)
        try:
            r = requests.get(
                f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool_id}/ohlcv/minute",
                headers=headers, timeout=15,
                params={"aggregate": "1", "limit": "200", "token": "base"}
            )
            if r.status_code != 200:
                return None

            ohlcv_list = r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
            if not ohlcv_list:
                return None

            bars = []
            for bar in ohlcv_list:
                if len(bar) >= 6:
                    bars.append({
                        "timestamp": int(bar[0]),
                        "open": float(bar[1]), "high": float(bar[2]),
                        "low": float(bar[3]), "close": float(bar[4]),
                        "volume": float(bar[5]),
                    })

            bars.sort(key=lambda x: x["timestamp"])
            df = pd.DataFrame(bars)
            df.index = pd.to_datetime(df["timestamp"], unit="s")
            df = df[["open", "high", "low", "close", "volume"]]

            print(f"   🌐 GeckoTerminal: {len(df)} bars ({pool_name})")
            return df

        except Exception:
            return None

    def _read_klines_from_db(self, token_ca: str, db_path: str) -> pd.DataFrame:
        """从 SQLite 数据库读取 K 线"""
        import sqlite3

        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row

            rows = conn.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM kline_1m
                WHERE token_ca = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (token_ca, KRONOS_MAX_CONTEXT)).fetchall()
            conn.close()

            if len(rows) < 5:
                return None

            rows = list(reversed(rows))
            df = pd.DataFrame([dict(r) for r in rows])
            ts_col = df["timestamp"].iloc[0]
            unit = "s" if ts_col < 1e12 else "ms"
            df.index = pd.to_datetime(df["timestamp"], unit=unit)
            df = df[["open", "high", "low", "close", "volume"]]
            return df

        except Exception as e:
            print(f"   ⚠️ 读取 K 线失败: {e}")
            return None

    def _fallback_prediction(self) -> dict:
        """无数据时的降级输出"""
        return {
            "trend_direction": 0.0,
            "trend_magnitude": 0.0,
            "implied_volatility": 0.05,
            "confidence": 0.0,
            "pred_df": None,
            "upside": 0.0,
            "downside": 0.0,
        }
