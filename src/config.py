"""
Kronos Shadow Trader — 配置管理

通过 HTTP API 从现有系统获取数据，影子交易写入独立 DB。
支持两种模式: HTTP API (Zeabur 部署) 或本地文件 (本地调试)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 现有系统 API 连接 (Zeabur 部署时使用)
# ============================================================
SENTINEL_API_URL = os.getenv(
    "SENTINEL_API_URL",
    "https://sentiment-arbitrage.zeabur.app"
)
SENTINEL_TOKEN = os.getenv("SENTINEL_TOKEN", "mytoken54321")

# 数据获取模式: "api" (通过 HTTP) 或 "local" (本地文件)
DATA_MODE = os.getenv("DATA_MODE", "api")

# ============================================================
# 本地文件路径 (DATA_MODE=local 时使用)
# ============================================================
SENTIMENT_DB_PATH = os.getenv(
    "SENTIMENT_DB_PATH",
    str(Path(__file__).parent.parent.parent / "sentiment-arbitrage-system" / "server_sentiment_arb.db")
)
PAPER_DB_PATH = os.getenv(
    "PAPER_DB_PATH",
    str(Path(__file__).parent.parent.parent / "sentiment-arbitrage-system" / "server_paper.db")
)
KLINE_DB_PATH = os.getenv(
    "KLINE_DB_PATH",
    str(Path(__file__).parent.parent.parent / "sentiment-arbitrage-system" / "data" / "kline_cache.db")
)

# ============================================================
# 影子系统数据库 (读写)
# ============================================================
SHADOW_DB_PATH = os.getenv(
    "SHADOW_DB_PATH",
    str(Path(__file__).parent.parent / "data" / "shadow_trades.db")
)

# ============================================================
# Kronos 模型配置
# ============================================================
KRONOS_TOKENIZER = os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_MODEL = os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
KRONOS_DEVICE = os.getenv("KRONOS_DEVICE", "auto")  # auto / cpu / cuda / mps
KRONOS_MAX_CONTEXT = 512
KRONOS_PRED_LEN = 60         # 预测未来 60 根 1min K 线
KRONOS_SAMPLE_COUNT = 10     # 生成 10 条路径取平均

# ============================================================
# GBDT 模型配置
# ============================================================
GBDT_MODEL_PATH = os.getenv(
    "GBDT_MODEL_PATH",
    str(Path(__file__).parent.parent / "models" / "gbdt_model.pkl")
)
GBDT_MIN_TRAINING_SAMPLES = 100  # 最少需要多少条交易数据才能训练

# ============================================================
# 影子交易配置
# ============================================================
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))  # 轮询间隔 (秒)
SHADOW_POSITION_SIZE_SOL = 0.1    # 模拟仓位 (不实际执行)
SHADOW_STOP_LOSS_PCT = -35.0      # 止损百分比 (对标现有系统)
SHADOW_TRAIL_ACTIVATION_PCT = 5.0  # 追踪止损激活阈值

# ============================================================
# Kelly 配置 (对标现有 entry_engine.py 的安全机制)
# ============================================================
KELLY_DEFAULT_WIN_RATE = 0.30     # 默认胜率
KELLY_MAX_WIN_RATE_CAP = 0.65     # 胜率上限
KELLY_MIN_POSITION_SOL = 0.03     # 最小仓位
KELLY_MAX_POSITION_PCT = 0.20     # 最大仓位占总资金比
KELLY_ODDS_CAP = 3.0              # 赔率上限
KELLY_FRACTION = 0.5              # Half-Kelly

# ============================================================
# GBDT 特征列表 (从现有系统数据中提取)
# ============================================================
FEATURE_COLUMNS = [
    # 信号特征
    "market_cap",
    "holders",
    "volume_24h",
    "top10_pct",
    "is_ath",
    "signal_count",            # 同一 token 的信号数量
    "signal_velocity",         # 信号增长速度 (count/hour)
    # 价格特征
    "price_change_since_signal",  # 入场时价格 vs 信号价格
    "entry_delay_minutes",        # 信号到入场的延迟
    # K 线特征 (从 kline_cache 提取)
    "kline_bars_available",       # 可用 K 线数量
    "kline_trend_slope",          # 最近 N 根 K 线的趋势斜率
    "kline_volatility",           # 最近 N 根 K 线的波动率 (std/mean)
    "kline_volume_trend",         # 成交量趋势
    "kline_green_ratio",          # 阳线占比
    "kline_upper_wick_ratio",     # 上影线比率 (卖压信号)
    # 市场环境
    "market_regime",              # bull/bear/neutral (encoded)
]
