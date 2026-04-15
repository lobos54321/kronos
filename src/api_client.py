"""
现有系统 HTTP API 客户端

通过 HTTP 从现有 sentiment-arbitrage 系统获取信号和数据,
替代直接读取 SQLite 文件 (Zeabur 部署时两个服务不共享文件系统)。
"""

import json
import time
import requests
import sqlite3
import tempfile
import os
from pathlib import Path

from src.config import SENTINEL_API_URL, SENTINEL_TOKEN


class SentinelAPIClient:
    """
    从现有 sentiment-arbitrage 系统的 HTTP API 获取数据。
    
    可用接口:
    - /api/export?token=xxx            → premium_signals JSON (支持分页)
    - /api/download/database?token=xxx → sentiment_arb.db 下载
    - /api/download/kline_cache?token=xxx → kline_cache.db 下载
    - /api/download/paper_trades?token=xxx → paper_trades.db 下载
    """

    def __init__(self, base_url=None, token=None):
        self.base_url = (base_url or SENTINEL_API_URL).rstrip("/")
        self.token = token or SENTINEL_TOKEN
        self._cache_dir = Path(tempfile.gettempdir()) / "kronos_shadow_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._kline_db_path = None
        self._kline_db_last_download = 0

    def _get(self, path, params=None, stream=False, timeout=60):
        """发送 GET 请求"""
        url = f"{self.base_url}{path}"
        if params is None:
            params = {}
        params["token"] = self.token
        try:
            r = requests.get(url, params=params, stream=stream, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"⚠️  API 请求失败 [{path}]: {e}")
            return None

    def get_new_signals(self, after_id=0, limit=50) -> list:
        """
        获取 ID > after_id 的新信号。
        使用 /api/export 接口拉取 premium_signals 数据。
        """
        r = self._get("/api/export", params={"limit": str(limit)})
        if r is None:
            return []

        try:
            data = r.json()
            signals_data = data.get("tables", {}).get("premium_signals", {})
            rows = signals_data.get("rows", [])

            # 过滤出 id > after_id 的信号
            new_signals = []
            for row in rows:
                sig_id = row.get("id", 0)
                if sig_id > after_id:
                    new_signals.append({
                        "id": sig_id,
                        "token_ca": row.get("token_ca", ""),
                        "symbol": row.get("symbol", "?"),
                        "market_cap": row.get("market_cap", 0),
                        "holders": row.get("holders", 0),
                        "volume_24h": row.get("volume_24h", 0),
                        "top10_pct": row.get("top10_pct", 0),
                        "timestamp": row.get("timestamp") or row.get("ts") or int(time.time()),
                        "signal_type": row.get("signal_type", "premium"),
                        "is_ath": row.get("is_ath", 0),
                    })

            # 按 ID 升序排列
            new_signals.sort(key=lambda x: x["id"])
            return new_signals[:limit]

        except Exception as e:
            print(f"⚠️  解析导出数据失败: {e}")
            return []

    def count_prior_signals(self, token_ca: str, before_ts: int) -> dict:
        """
        统计某个 token 在 before_ts 之前的信号数量
        (从 export 数据中计算, 不需要额外接口)
        """
        r = self._get("/api/export", params={"limit": "1000"})
        if r is None:
            return {"signal_count": 1, "signal_velocity": 0}

        try:
            data = r.json()
            rows = data.get("tables", {}).get("premium_signals", {}).get("rows", [])

            matching = [
                row for row in rows
                if row.get("token_ca") == token_ca
                and (row.get("timestamp") or row.get("ts") or 0) < before_ts
            ]
            count = len(matching) + 1  # +1 包含当前信号

            # 计算信号速度 (count per hour)
            if len(matching) >= 2:
                timestamps = sorted([r.get("timestamp") or r.get("ts") or 0 for r in matching])
                time_span_hours = max((timestamps[-1] - timestamps[0]) / 3600, 0.1)
                velocity = len(matching) / time_span_hours
            else:
                velocity = 0

            return {"signal_count": count, "signal_velocity": velocity}
        except Exception:
            return {"signal_count": 1, "signal_velocity": 0}

    def download_kline_db(self, force=False) -> str:
        """
        下载 kline_cache.db 到本地缓存。
        每 5 分钟最多下载一次 (避免频繁下载)。
        返回本地文件路径。
        """
        cache_path = str(self._cache_dir / "kline_cache.db")
        now = time.time()

        # 5 分钟内不重新下载
        if not force and self._kline_db_path and (now - self._kline_db_last_download) < 300:
            if os.path.exists(cache_path):
                return cache_path

        print("📥 下载 K 线缓存数据库...")
        r = self._get("/api/download/kline_cache", stream=True, timeout=120)
        if r is None:
            # 返回已有缓存 (如果有)
            if os.path.exists(cache_path):
                print("   使用本地缓存")
                return cache_path
            return ""

        try:
            with open(cache_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"   ✅ K 线缓存下载完成: {size_mb:.1f} MB")
            self._kline_db_path = cache_path
            self._kline_db_last_download = now
            return cache_path
        except Exception as e:
            print(f"⚠️  下载 K 线缓存失败: {e}")
            return cache_path if os.path.exists(cache_path) else ""

    def get_kline_features(self, token_ca: str, before_ts: int) -> dict:
        """
        提取 K 线特征。
        优先从 kline_cache.db, 不够则从 GeckoTerminal 获取。
        """
        default_features = {
            "kline_bars_available": 0, "kline_trend_slope": 0,
            "kline_volatility": 0, "kline_volume_trend": 0,
            "kline_green_ratio": 0, "kline_upper_wick_ratio": 0,
        }

        # 策略 1: 从 kline_cache.db 提取
        db_path = self.download_kline_db()
        if db_path:
            try:
                from src.gbdt.feature_extractor import _extract_kline_features, _connect_readonly
                conn = _connect_readonly(db_path)
                features = _extract_kline_features(conn, token_ca, before_ts)
                conn.close()
                if features.get("kline_bars_available", 0) > 0:
                    return features
            except Exception:
                pass

        # 策略 2: kline_cache 没有 → 从 GeckoTerminal 获取后计算特征
        try:
            bars = self._fetch_gecko_bars(token_ca)
            if bars and len(bars) >= 5:
                return self._compute_kline_features(bars)
        except Exception:
            pass

        return default_features

    def _fetch_gecko_bars(self, token_ca: str) -> list:
        """从 GeckoTerminal 获取 K 线原始数据"""
        import time as _time

        headers = {"Accept": "application/json", "User-Agent": "KronosShadow/1.0"}

        try:
            r = requests.get(
                f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_ca}/pools",
                headers=headers, timeout=15,
                params={"page": "1", "sort": "h24_volume_usd_desc"}
            )
            if r.status_code != 200:
                return []
            pools = r.json().get("data", [])
            if not pools:
                return []

            pool_id = pools[0].get("id", "").replace("solana_", "")
            _time.sleep(2)

            r = requests.get(
                f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool_id}/ohlcv/minute",
                headers=headers, timeout=15,
                params={"aggregate": "1", "limit": "200", "token": "base"}
            )
            if r.status_code != 200:
                return []

            ohlcv = r.json().get("data", {}).get("attributes", {}).get("ohlcv_list", [])
            bars = []
            for bar in ohlcv:
                if len(bar) >= 6:
                    bars.append({
                        "timestamp": int(bar[0]),
                        "open": float(bar[1]), "high": float(bar[2]),
                        "low": float(bar[3]), "close": float(bar[4]),
                        "volume": float(bar[5]),
                    })
            bars.sort(key=lambda x: x["timestamp"])
            return bars
        except Exception:
            return []

    def _compute_kline_features(self, bars: list) -> dict:
        """从原始 K 线数组计算 GBDT 特征"""
        import numpy as np

        n = len(bars)
        closes = [b["close"] for b in bars]
        opens = [b["open"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        volumes = [b["volume"] for b in bars]

        # 趋势斜率 (线性回归)
        x = np.arange(n)
        if n >= 2 and np.std(closes) > 0:
            slope = float(np.polyfit(x, closes, 1)[0])
            slope_normalized = slope / (np.mean(closes) + 1e-15)
        else:
            slope_normalized = 0.0

        # 波动率
        mean_close = np.mean(closes) if closes else 1
        volatility = float(np.std(closes) / (mean_close + 1e-15))

        # 成交量趋势
        if n >= 2 and np.std(volumes) > 0:
            vol_slope = float(np.polyfit(x, volumes, 1)[0])
            vol_normalized = vol_slope / (np.mean(volumes) + 1e-15)
        else:
            vol_normalized = 0.0

        # 阳线占比
        green_count = sum(1 for o, c in zip(opens, closes) if c >= o)
        green_ratio = green_count / n if n > 0 else 0.5

        # 上影线比率
        wick_ratios = []
        for h, c, o in zip(highs, closes, opens):
            body_top = max(c, o)
            bar_range = h - min(c, o)
            if bar_range > 0:
                wick_ratios.append((h - body_top) / bar_range)
        upper_wick = float(np.mean(wick_ratios)) if wick_ratios else 0.0

        return {
            "kline_bars_available": n,
            "kline_trend_slope": slope_normalized,
            "kline_volatility": volatility,
            "kline_volume_trend": vol_normalized,
            "kline_green_ratio": green_ratio,
            "kline_upper_wick_ratio": upper_wick,
        }

