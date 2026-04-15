"""
GBDT 特征提取器

从现有系统的数据库中 **只读** 提取特征，构建训练数据集。
数据来源:
  - server_paper.db          → 已关闭的交易记录 (标签 Y)
  - server_sentiment_arb.db  → premium_signals (特征 X)
  - kline_cache.db           → 1min K 线 (价格特征)
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    SENTIMENT_DB_PATH, PAPER_DB_PATH, KLINE_DB_PATH,
    FEATURE_COLUMNS, GBDT_MIN_TRAINING_SAMPLES
)


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    """以只读模式连接数据库，绝不写入"""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _extract_kline_features(kline_conn: sqlite3.Connection, token_ca: str,
                             before_ts: int, lookback_bars: int = 60) -> dict:
    """从 kline_cache 提取指定 token 在 before_ts 之前的 K 线特征"""
    rows = kline_conn.execute("""
        SELECT timestamp, open, high, low, close, volume
        FROM kline_1m
        WHERE token_ca = ? AND timestamp < ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (token_ca, before_ts, lookback_bars)).fetchall()

    if len(rows) < 5:
        return {
            "kline_bars_available": len(rows),
            "kline_trend_slope": 0.0,
            "kline_volatility": 0.0,
            "kline_volume_trend": 0.0,
            "kline_green_ratio": 0.0,
            "kline_upper_wick_ratio": 0.0,
        }

    # 按时间正序
    rows = list(reversed(rows))
    closes = np.array([r["close"] for r in rows], dtype=np.float64)
    opens = np.array([r["open"] for r in rows], dtype=np.float64)
    highs = np.array([r["high"] for r in rows], dtype=np.float64)
    lows = np.array([r["low"] for r in rows], dtype=np.float64)
    volumes = np.array([r["volume"] for r in rows], dtype=np.float64)

    n = len(closes)

    # 趋势斜率: 线性回归 close 的 slope (标准化)
    x = np.arange(n, dtype=np.float64)
    mean_close = np.mean(closes)
    if mean_close > 0:
        slope = np.polyfit(x, closes, 1)[0] / mean_close  # 标准化
    else:
        slope = 0.0

    # 波动率: std(returns)
    returns = np.diff(closes) / (closes[:-1] + 1e-15)
    volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

    # 成交量趋势
    if len(volumes) > 1 and np.mean(volumes) > 0:
        vol_slope = np.polyfit(x, volumes, 1)[0] / np.mean(volumes)
    else:
        vol_slope = 0.0

    # 阳线比率
    green_count = np.sum(closes >= opens)
    green_ratio = float(green_count / n)

    # 上影线比率 (卖压)
    candle_range = highs - lows + 1e-15
    upper_wick = highs - np.maximum(closes, opens)
    upper_wick_ratio = float(np.mean(upper_wick / candle_range))

    return {
        "kline_bars_available": n,
        "kline_trend_slope": float(slope),
        "kline_volatility": volatility,
        "kline_volume_trend": float(vol_slope),
        "kline_green_ratio": green_ratio,
        "kline_upper_wick_ratio": upper_wick_ratio,
    }


def _count_prior_signals(sig_conn: sqlite3.Connection, token_ca: str,
                          before_ts: int) -> dict:
    """统计该 token 在入场前收到的信号数量和速度"""
    rows = sig_conn.execute("""
        SELECT timestamp FROM premium_signals
        WHERE token_ca = ? AND timestamp < ?
        ORDER BY timestamp ASC
    """, (token_ca, before_ts)).fetchall()

    count = len(rows)
    velocity = 0.0
    if count >= 2:
        ts_list = [r["timestamp"] for r in rows]
        span_hours = (ts_list[-1] - ts_list[0]) / 3600000  # ms → hours
        if span_hours > 0:
            velocity = count / span_hours

    return {
        "signal_count": count,
        "signal_velocity": velocity,
    }


def extract_training_data() -> pd.DataFrame:
    """
    从现有数据库提取完整的训练数据集。

    Returns:
        DataFrame with feature columns + target columns (pnl_pct, is_winner)
    """
    paper_conn = _connect_readonly(PAPER_DB_PATH)
    sig_conn = _connect_readonly(SENTIMENT_DB_PATH)
    kline_conn = _connect_readonly(KLINE_DB_PATH)

    # 获取所有已关闭的 paper trades
    trades = paper_conn.execute("""
        SELECT id, token_ca, symbol, signal_ts, entry_price, entry_ts,
               exit_price, exit_ts, exit_reason, pnl_pct, peak_pnl,
               bars_held, market_regime, signal_type
        FROM paper_trades
        WHERE pnl_pct IS NOT NULL
        ORDER BY entry_ts ASC
    """).fetchall()

    print(f"📊 从 paper_trades 读取 {len(trades)} 条已关闭交易")

    records = []
    for i, trade in enumerate(trades):
        token_ca = trade["token_ca"]
        signal_ts = trade["signal_ts"]
        entry_ts = trade["entry_ts"]

        # --- 从 premium_signals 拿特征 ---
        signal = sig_conn.execute("""
            SELECT market_cap, holders, volume_24h, top10_pct, is_ath
            FROM premium_signals
            WHERE token_ca = ? AND timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1
        """, (token_ca, entry_ts)).fetchone()

        if signal:
            market_cap = signal["market_cap"] or 0
            holders = signal["holders"] or 0
            volume_24h = signal["volume_24h"] or 0
            top10_pct = signal["top10_pct"] or 0
            is_ath = signal["is_ath"] or 0
        else:
            market_cap, holders, volume_24h, top10_pct, is_ath = 0, 0, 0, 0, 0

        # --- 信号统计特征 ---
        sig_features = _count_prior_signals(sig_conn, token_ca, entry_ts)

        # --- K 线特征 ---
        kline_features = _extract_kline_features(kline_conn, token_ca, entry_ts)

        # --- 价格变化特征 ---
        entry_price = trade["entry_price"]
        # 用第一个信号的时间点价格估算 (signal_ts 对应的 entry_price 近似)
        price_change_since_signal = 0.0  # 无法精确计算，留 0

        # 入场延迟
        entry_delay_ms = entry_ts - signal_ts if signal_ts and entry_ts else 0
        entry_delay_minutes = max(0, entry_delay_ms / 60000)

        # --- 市场环境编码 ---
        regime_map = {"bull": 1.0, "neutral": 0.0, "bear": -1.0}
        market_regime = regime_map.get(trade["market_regime"], 0.0)

        # --- 组装记录 ---
        record = {
            # 特征 X
            "market_cap": market_cap,
            "holders": holders,
            "volume_24h": volume_24h,
            "top10_pct": top10_pct,
            "is_ath": is_ath,
            "signal_count": sig_features["signal_count"],
            "signal_velocity": sig_features["signal_velocity"],
            "price_change_since_signal": price_change_since_signal,
            "entry_delay_minutes": entry_delay_minutes,
            **kline_features,
            "market_regime": market_regime,

            # 标签 Y
            "pnl_pct": trade["pnl_pct"],
            "is_winner": 1 if (trade["pnl_pct"] or 0) > 0 else 0,
            "peak_pnl": trade["peak_pnl"] or 0,

            # 元数据 (不参与训练)
            "_trade_id": trade["id"],
            "_token_ca": token_ca,
            "_symbol": trade["symbol"],
            "_signal_type": trade["signal_type"],
            "_exit_reason": trade["exit_reason"],
            "_entry_ts": entry_ts,
        }
        records.append(record)

        if (i + 1) % 200 == 0:
            print(f"  处理进度: {i + 1}/{len(trades)}")

    paper_conn.close()
    sig_conn.close()
    kline_conn.close()

    df = pd.DataFrame(records)
    print(f"✅ 提取完成: {len(df)} 条记录, {df['is_winner'].sum()} 条盈利 ({df['is_winner'].mean()*100:.1f}%)")
    return df


if __name__ == "__main__":
    df = extract_training_data()
    # 保存到 CSV 供检查
    out_path = Path(__file__).parent.parent.parent / "data" / "training_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"💾 训练数据已保存到 {out_path}")
    print(f"\n📈 特征统计:")
    print(df.describe().to_string())
