"""
Kronos K 线预测引擎

封装 Kronos Foundation Model 的推理逻辑。
输入历史 1min K 线 → 输出未来 K 线预测 + 趋势方向 + 波动率。

K 线数据获取策略 (按优先级):
1. 本地缓存 kline_cache.db (如果有)
2. 现有系统 API 下载 kline_cache.db
3. 直接调 GeckoTerminal API 获取 (公开免费, 无需 key)
"""

import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    KRONOS_TOKENIZER, KRONOS_MODEL, KRONOS_DEVICE,
    KRONOS_MAX_CONTEXT, KRONOS_PRED_LEN, KRONOS_SAMPLE_COUNT,
    KLINE_DB_PATH, DATA_MODE,
)


def fetch_klines_from_gecko(token_ca: str, limit: int = 200) -> pd.DataFrame:
    """
    直接从 GeckoTerminal API 获取 K 线数据。
    
    流程:
    1. 用 token_ca 查询 pool 地址
    2. 用 pool 地址获取 OHLCV 数据
    
    GeckoTerminal API 是公开的, 不需要 API key。
    限流: ~30 req/min, 我们控制在 1 req/5s。
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "KronosShadow/1.0"
    }

    # Step 1: 查找 token 的 pool 地址
    try:
        pool_url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_ca}/pools"
        r = requests.get(pool_url, headers=headers, timeout=15,
                         params={"page": "1", "sort": "h24_volume_usd_desc"})
        if r.status_code == 429:
            print(f"   ⚠️ GeckoTerminal 限流, 等待 30s...")
            time.sleep(30)
            r = requests.get(pool_url, headers=headers, timeout=15,
                             params={"page": "1", "sort": "h24_volume_usd_desc"})
        if r.status_code != 200:
            print(f"   ⚠️ GeckoTerminal pool 查询失败: HTTP {r.status_code}")
            return pd.DataFrame()

        pools = r.json().get("data", [])
        if not pools:
            print(f"   ⚠️ 未找到 {token_ca[:12]}... 的流动池")
            return pd.DataFrame()

        pool_address = pools[0].get("attributes", {}).get("address", "")
        if not pool_address:
            pool_address = pools[0].get("id", "").replace("solana_", "")
        
        if not pool_address:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"   ⚠️ GeckoTerminal pool 查询异常: {e}")
        return pd.DataFrame()

    # Step 2: 获取 OHLCV K 线
    time.sleep(2)  # 避免限流
    try:
        all_bars = []
        now_ts = int(time.time())
        
        # GeckoTerminal 每次最多 1000 条, 我们分批获取
        # 对于 512 根 1min K 线, 一次就够了
        for batch in range(3):  # 最多 3 批, 共 ~600 条
            before_ts = now_ts - batch * 200 * 60
            ohlcv_url = (
                f"https://api.geckoterminal.com/api/v2/networks/solana/"
                f"pools/{pool_address}/ohlcv/minute"
            )
            r = requests.get(ohlcv_url, headers=headers, timeout=15, params={
                "aggregate": "1",
                "limit": "200",
                "before_timestamp": str(before_ts),
                "token": "base",
            })
            
            if r.status_code == 429:
                print(f"   ⚠️ GeckoTerminal 限流, 停止获取")
                break
            if r.status_code != 200:
                break

            ohlcv_list = r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
            if not ohlcv_list:
                break

            for bar in ohlcv_list:
                if len(bar) >= 6:
                    all_bars.append({
                        "timestamp": int(bar[0]),
                        "open": float(bar[1]),
                        "high": float(bar[2]),
                        "low": float(bar[3]),
                        "close": float(bar[4]),
                        "volume": float(bar[5]),
                    })

            if len(ohlcv_list) < 200:
                break  # 没有更多数据了
            
            time.sleep(2)  # 批次间等待

        if not all_bars:
            print(f"   ⚠️ 未获取到 K 线数据")
            return pd.DataFrame()

        # 去重, 按时间排序
        seen = set()
        unique_bars = []
        for bar in all_bars:
            if bar["timestamp"] not in seen:
                seen.add(bar["timestamp"])
                unique_bars.append(bar)
        unique_bars.sort(key=lambda x: x["timestamp"])

        df = pd.DataFrame(unique_bars)
        df.index = pd.to_datetime(df["timestamp"], unit="s")
        df = df[["open", "high", "low", "close", "volume"]]

        print(f"   📊 GeckoTerminal 获取 {len(df)} 根 K 线 ({pools[0].get('attributes', {}).get('name', 'unknown')})")
        return df

    except Exception as e:
        print(f"   ⚠️ GeckoTerminal OHLCV 获取异常: {e}")
        return pd.DataFrame()


class KronosPredictor:
    """
    Kronos K 线预测器。

    对单个 token 的历史 K 线做预测, 输出:
    - 未来 N 根 K 线的 OHLCV (直接用于趋势判断)
    - 隐含波动率 (high-low spread)
    - 预测置信度 (多路径方差)
    """

    def __init__(self, device: str = None):
        self.device = device or KRONOS_DEVICE
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self._loaded = False

    def load(self) -> bool:
        """加载 Kronos 模型 (首次会从 HuggingFace 下载)"""
        try:
            import torch

            # 延迟导入 Kronos 模块
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
            print(f"   请先下载 Kronos 源码到 src/kronos/kronos_model.py")
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

        Returns:
            {
                "trend_direction": float,    # >0 看涨, <0 看跌
                "trend_magnitude": float,    # 趋势幅度 (%)
                "implied_volatility": float, # 隐含波动率
                "confidence": float,         # 预测置信度 (0-1)
                "pred_df": DataFrame,        # 完整预测结果
                "upside": float,             # 预期上行幅度
                "downside": float,           # 预期下行幅度
            }
        """
        if not self._loaded:
            if not self.load():
                return self._fallback_prediction()

        pred_len = pred_len or KRONOS_PRED_LEN

        # 确保列名正确
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
            # 假设 index 是 unix timestamp (ms)
            x_timestamp = pd.to_datetime(kline_df.index, unit='ms')

        # 构造未来时间戳 (每分钟)
        last_ts = x_timestamp[-1]
        y_timestamp = pd.date_range(
            start=last_ts + timedelta(minutes=1),
            periods=pred_len,
            freq='1min'
        )

        try:
            # 多路径采样预测
            pred_df = self.predictor.predict(
                df=df.reset_index(drop=True),
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=KRONOS_SAMPLE_COUNT,
            )

            # 分析预测结果
            current_price = float(df["close"].iloc[-1])
            pred_closes = pred_df["close"].values
            pred_highs = pred_df["high"].values
            pred_lows = pred_df["low"].values

            # 趋势方向和幅度
            future_avg_close = np.mean(pred_closes)
            trend_direction = (future_avg_close - current_price) / current_price
            trend_magnitude = abs(trend_direction) * 100

            # 隐含波动率
            pred_ranges = pred_highs - pred_lows
            implied_vol = float(np.mean(pred_ranges / (pred_closes + 1e-15)))

            # 上行/下行估算
            upside = float((np.max(pred_highs) - current_price) / current_price)
            downside = float((current_price - np.min(pred_lows)) / current_price)

            # 置信度 (基于预测的稳定性 — 趋势一致性)
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
        
        数据获取策略 (按优先级):
        1. 本地/缓存的 kline_cache.db
        2. 直接从 GeckoTerminal API 获取
        """
        kline_df = self._get_klines(token_ca)

        if kline_df is None or len(kline_df) < 10:
            return self._fallback_prediction()

        return self.predict_from_klines(kline_df, pred_len)

    def _get_klines(self, token_ca: str) -> pd.DataFrame:
        """
        多源获取 K 线数据。
        优先本地缓存, 不够则从 GeckoTerminal 补充。
        """
        df = None

        # 策略 1: 尝试从本地/缓存 DB 读取
        if DATA_MODE == "local":
            df = self._read_klines_from_db(token_ca)
        elif DATA_MODE == "api":
            # 尝试从 API 下载的缓存 DB 读取
            try:
                from src.api_client import SentinelAPIClient
                client = SentinelAPIClient()
                db_path = client.download_kline_db()
                if db_path:
                    df = self._read_klines_from_db(token_ca, db_path)
            except Exception:
                pass

        # 如果本地数据足够, 直接用
        if df is not None and len(df) >= 60:
            return df

        # 策略 2: 从 GeckoTerminal API 直接获取
        print(f"   🌐 从 GeckoTerminal 获取 K 线...")
        gecko_df = fetch_klines_from_gecko(token_ca, limit=KRONOS_MAX_CONTEXT)

        if gecko_df is not None and len(gecko_df) >= 10:
            # 如果有本地数据, 合并 (优先用本地的, GeckoTerminal 补充历史)
            if df is not None and len(df) > 0:
                combined = pd.concat([gecko_df, df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                return combined.tail(KRONOS_MAX_CONTEXT)
            return gecko_df.tail(KRONOS_MAX_CONTEXT)

        # 如果都没有, 返回本地有多少就用多少
        return df

    def _read_klines_from_db(self, token_ca: str, db_path: str = None) -> pd.DataFrame:
        """从 SQLite 数据库读取 K 线"""
        import sqlite3
        path = db_path or KLINE_DB_PATH

        try:
            uri = f"file:{path}?mode=ro"
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
            print(f"   ⚠️ 读取本地 K 线失败: {e}")
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
