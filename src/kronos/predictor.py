"""
Kronos K 线预测引擎

封装 Kronos Foundation Model 的推理逻辑。
输入历史 1min K 线 → 输出未来 K 线预测 + 趋势方向 + 波动率。

K 线数据来源:
- 现有系统的 kline_cache.db (通过 /api/download/kline_cache 下载)
- 现有系统已有完整的 K 线管道 (KlineCollector + GeckoTerminal + Helius)
- 影子系统只做消费者, 不重复获取
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    KRONOS_TOKENIZER, KRONOS_MODEL, KRONOS_DEVICE,
    KRONOS_MAX_CONTEXT, KRONOS_PRED_LEN, KRONOS_SAMPLE_COUNT,
    KLINE_DB_PATH, DATA_MODE,
)


class KronosPredictor:
    """
    Kronos K 线预测器。

    对单个 token 的历史 K 线做预测, 输出:
    - 未来 N 根 K 线的 OHLCV (直接用于趋势判断)
    - 隐含波动率 (high-low spread)
    - 预测置信度 (多路径方差)
    """

    def __init__(self, device: str = None, api_client=None):
        self.device = device or KRONOS_DEVICE
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self._loaded = False
        self._api_client = api_client  # 共享的 API 客户端实例

    def load(self) -> bool:
        """加载 Kronos 模型 (首次会从 HuggingFace 下载)"""
        try:
            import torch

            from src.kronos.kronos_model import Kronos, KronosTokenizer, KronosPredictor as KP

            print(f"⏳ 正在加载 Kronos 模型...")
            print(f"   Tokenizer: {KRONOS_TOKENIZER}")
            print(f"   Model: {KRONOS_MODEL}")

            self.tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER)
            self.model = Kronos.from_pretrained(KRONOS_MODEL)

            # 自动选择设备
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            self.predictor = KP(
                self.model, self.tokenizer,
                device=self.device,
                max_context=KRONOS_MAX_CONTEXT,
            )

            self._loaded = True
            print(f"✅ Kronos 模型已加载 (device: {self.device})")
            return True

        except ImportError as e:
            print(f"⚠️  Kronos 模型导入失败: {e}")
            return False
        except Exception as e:
            print(f"❌ Kronos 模型加载失败: {e}")
            return False

    def predict_from_klines(self, kline_df: pd.DataFrame,
                             pred_len: int = None) -> dict:
        """
        从历史 K 线 DataFrame 预测未来走势。

        Args:
            kline_df: DataFrame with columns [open, high, low, close, volume]
                      按时间正序排列, index 为 timestamp
            pred_len: 预测长度 (根 K 线), 默认用配置值
        """
        if not self._loaded:
            if not self.load():
                return self._fallback_prediction()

        pred_len = pred_len or KRONOS_PRED_LEN

        required_cols = ["open", "high", "low", "close"]
        if not all(c in kline_df.columns for c in required_cols):
            print(f"⚠️  K 线数据缺少必要列: {required_cols}")
            return self._fallback_prediction()

        if len(kline_df) < 10:
            print(f"⚠️  K 线数据太少: {len(kline_df)} 根 (最少 10 根)")
            return self._fallback_prediction()

        # 准备输入数据
        df = kline_df[["open", "high", "low", "close"]].copy()
        if "volume" in kline_df.columns:
            df["volume"] = kline_df["volume"]
        else:
            df["volume"] = 0.0
        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df["close"]

        # 构造时间戳
        if isinstance(kline_df.index, pd.DatetimeIndex):
            x_timestamp = kline_df.index
        else:
            x_timestamp = pd.to_datetime(kline_df.index, unit='ms')

        # 构造未来时间戳 (每分钟)
        last_ts = x_timestamp[-1]
        y_timestamp = pd.date_range(
            start=last_ts + timedelta(minutes=1),
            periods=pred_len,
            freq='1min'
        )

        try:
            pred_df = self.predictor.predict(
                df=df.reset_index(drop=True),
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=KRONOS_SAMPLE_COUNT,
            )

            current_price = float(df["close"].iloc[-1])
            pred_closes = pred_df["close"].values
            pred_highs = pred_df["high"].values
            pred_lows = pred_df["low"].values

            future_avg_close = np.mean(pred_closes)
            trend_direction = (future_avg_close - current_price) / current_price
            trend_magnitude = abs(trend_direction) * 100

            pred_ranges = pred_highs - pred_lows
            implied_vol = float(np.mean(pred_ranges / (pred_closes + 1e-15)))

            upside = float((np.max(pred_highs) - current_price) / current_price)
            downside = float((current_price - np.min(pred_lows)) / current_price)

            returns = np.diff(pred_closes) / (pred_closes[:-1] + 1e-15)
            if trend_direction > 0:
                confidence = float(np.mean(returns > 0))
            else:
                confidence = float(np.mean(returns < 0))

            return {
                "trend_direction": float(trend_direction),
                "trend_magnitude": float(trend_magnitude),
                "implied_volatility": float(implied_vol),
                "confidence": float(confidence),
                "pred_df": pred_df,
                "upside": max(0, upside),
                "downside": max(0, downside),
            }

        except Exception as e:
            print(f"❌ Kronos 预测失败: {e}")
            return self._fallback_prediction()

    def predict_for_token(self, token_ca: str, pred_len: int = None) -> dict:
        """
        获取指定 token 的 K 线并预测。
        
        K 线获取策略:
        1. 现有系统的 kline_cache.db (已有 ~1800 token 的数据)
        2. GeckoTerminal 免费 API (补充 kline_cache 未覆盖的新 token)
        """
        kline_df = self._get_klines(token_ca)

        if kline_df is None or len(kline_df) < 10:
            return self._fallback_prediction()

        return self.predict_from_klines(kline_df, pred_len)

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

        # 策略 2: kline_cache 没有数据 → 从 GeckoTerminal 获取
        # (现有系统的 KlineCollector 只收集已在监控列表中的 token,
        #  新信号的 meme coin 通常还没被收集)
        return self._fetch_from_gecko(token_ca)

    def _fetch_from_gecko(self, token_ca: str) -> pd.DataFrame:
        """从 GeckoTerminal 免费 API 获取 K 线 (补充 kline_cache 盲区)"""
        import time
        import requests

        headers = {"Accept": "application/json", "User-Agent": "KronosShadow/1.0"}

        # Step 1: 查找 pool 地址
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

        # Step 2: 获取 OHLCV
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

            if len(rows) < 10:
                return None

            # 转成 DataFrame (正序)
            rows = list(reversed(rows))
            df = pd.DataFrame([dict(r) for r in rows])
            # timestamp 可能是秒或毫秒, 智能判断
            ts_col = df["timestamp"].iloc[0]
            unit = "s" if ts_col < 1e12 else "ms"
            df.index = pd.to_datetime(df["timestamp"], unit=unit)
            df = df[["open", "high", "low", "close", "volume"]]
            return df

        except Exception as e:
            print(f"   ⚠️ 读取 K 线失败: {e}")
            return None

    def _fallback_prediction(self) -> dict:
        """模型不可用时的降级输出"""
        return {
            "trend_direction": 0.0,
            "trend_magnitude": 0.0,
            "implied_volatility": 0.05,
            "confidence": 0.0,
            "pred_df": None,
            "upside": 0.0,
            "downside": 0.0,
        }
