"""
影子交易主循环

轮询现有系统的 premium_signals，用 GBDT + Kronos + Kelly 做影子决策。
所有决策记录到独立数据库，不执行任何真实交易。

支持两种数据获取模式:
- DATA_MODE=api  → 通过 HTTP API 从现有系统拉取 (Zeabur 部署)
- DATA_MODE=local → 直接读取本地 SQLite 文件 (本地调试)
"""

import sqlite3
import time
import json
import signal as signal_module
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.config import (
    SENTINEL_API_URL, SENTINEL_TOKEN, DATA_MODE,
    SENTIMENT_DB_PATH, KLINE_DB_PATH, SHADOW_DB_PATH,
    POLL_INTERVAL, SHADOW_POSITION_SIZE_SOL,
)
from src.gbdt.predictor import GBDTPredictor
from src.kronos.predictor import KronosPredictor
from src.kelly.enhanced_kelly import EnhancedKelly


class ShadowTrader:
    """
    影子交易器 — 只看信号、做判断、记录结果，不执行任何操作。
    """

    def __init__(self):
        self._api_client = None

        # 根据模式初始化数据源
        if DATA_MODE == "api":
            from src.api_client import SentinelAPIClient
            self._api_client = SentinelAPIClient()
            print(f"📡 数据模式: HTTP API ({SENTINEL_API_URL})")
        else:
            print(f"📂 数据模式: 本地文件")

        self.gbdt = GBDTPredictor()
        self.kronos = KronosPredictor(api_client=self._api_client)
        self.kelly = EnhancedKelly(total_capital_sol=5.0)
        self.running = False
        self._last_signal_id = 0

    def init_shadow_db(self):
        """初始化影子交易数据库"""
        db_dir = Path(SHADOW_DB_PATH).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(SHADOW_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                token_ca TEXT NOT NULL,
                symbol TEXT,
                signal_ts INTEGER,

                -- GBDT 输出
                gbdt_win_prob REAL,
                gbdt_expected_pnl REAL,
                gbdt_confidence TEXT,

                -- Kronos 输出
                kronos_trend_direction REAL,
                kronos_trend_magnitude REAL,
                kronos_implied_volatility REAL,
                kronos_confidence REAL,
                kronos_upside REAL,
                kronos_downside REAL,

                -- Kelly 输出
                kelly_decision TEXT,
                kelly_position_sol REAL,
                kelly_win_prob REAL,
                kelly_odds REAL,
                kelly_fraction REAL,
                kelly_reasoning TEXT,

                -- 信号原始数据
                signal_market_cap REAL,
                signal_holders INTEGER,
                signal_type TEXT,

                -- 跟踪: 用来事后对比
                entry_price_at_decision REAL,
                actual_price_30min REAL,
                actual_price_60min REAL,
                actual_price_120min REAL,
                actual_pnl_30min REAL,
                actual_pnl_60min REAL,
                actual_pnl_120min REAL,

                -- 现有系统的决策 (对比用)
                existing_system_action TEXT,
                existing_system_pnl REAL,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(signal_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_price_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER NOT NULL,
                token_ca TEXT NOT NULL,
                check_ts INTEGER,
                price REAL,
                minutes_since_decision INTEGER,
                FOREIGN KEY (decision_id) REFERENCES shadow_decisions(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        print(f"✅ 影子数据库已初始化: {SHADOW_DB_PATH}")

    def _get_last_processed_signal_id(self) -> int:
        """获取最后处理的信号 ID"""
        try:
            conn = sqlite3.connect(SHADOW_DB_PATH)
            row = conn.execute(
                "SELECT value FROM shadow_system_state WHERE key='last_signal_id'"
            ).fetchone()
            conn.close()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _save_last_processed_signal_id(self, signal_id: int):
        """保存最后处理的信号 ID"""
        conn = sqlite3.connect(SHADOW_DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO shadow_system_state (key, value, updated_at)
            VALUES ('last_signal_id', ?, datetime('now'))
        """, (str(signal_id),))
        conn.commit()
        conn.close()

    def _get_new_signals(self) -> list:
        """获取新信号 — 自动选择 API 或本地模式"""
        if self._api_client:
            return self._get_new_signals_api()
        else:
            return self._get_new_signals_local()

    def _get_new_signals_api(self) -> list:
        """通过 HTTP API 获取新信号"""
        try:
            return self._api_client.get_new_signals(
                after_id=self._last_signal_id, limit=50
            )
        except Exception as e:
            print(f"⚠️  API 读取信号失败: {e}")
            return []

    def _get_new_signals_local(self) -> list:
        """从本地 SQLite 文件获取新信号"""
        try:
            from src.gbdt.feature_extractor import _connect_readonly
            sig_conn = _connect_readonly(SENTIMENT_DB_PATH)
            rows = sig_conn.execute("""
                SELECT id, token_ca, symbol, market_cap, holders,
                       volume_24h, top10_pct, timestamp, signal_type, is_ath
                FROM premium_signals
                WHERE id > ?
                ORDER BY id ASC
                LIMIT 50
            """, (self._last_signal_id,)).fetchall()
            sig_conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"⚠️  读取信号失败: {e}")
            return []

    def _extract_features_for_signal(self, signal: dict) -> dict:
        """为单个信号提取特征"""
        token_ca = signal["token_ca"]
        ts = signal["timestamp"]

        # K 线特征
        if self._api_client:
            kline_features = self._api_client.get_kline_features(token_ca, ts)
        else:
            try:
                from src.gbdt.feature_extractor import _connect_readonly, _extract_kline_features
                kline_conn = _connect_readonly(KLINE_DB_PATH)
                kline_features = _extract_kline_features(kline_conn, token_ca, ts)
                kline_conn.close()
            except Exception:
                kline_features = {
                    "kline_bars_available": 0, "kline_trend_slope": 0,
                    "kline_volatility": 0, "kline_volume_trend": 0,
                    "kline_green_ratio": 0, "kline_upper_wick_ratio": 0,
                }

        # 信号统计
        if self._api_client:
            sig_features = self._api_client.count_prior_signals(token_ca, ts)
        else:
            try:
                from src.gbdt.feature_extractor import _connect_readonly, _count_prior_signals
                sig_conn = _connect_readonly(SENTIMENT_DB_PATH)
                sig_features = _count_prior_signals(sig_conn, token_ca, ts)
                sig_conn.close()
            except Exception:
                sig_features = {"signal_count": 1, "signal_velocity": 0}

        features = {
            "market_cap": signal.get("market_cap", 0) or 0,
            "holders": signal.get("holders", 0) or 0,
            "volume_24h": signal.get("volume_24h", 0) or 0,
            "top10_pct": signal.get("top10_pct", 0) or 0,
            "is_ath": signal.get("is_ath", 0) or 0,
            "signal_count": sig_features["signal_count"],
            "signal_velocity": sig_features["signal_velocity"],
            "price_change_since_signal": 0.0,
            "entry_delay_minutes": 0.0,
            **kline_features,
            "market_regime": 0.0,
        }

        # Debug: 打印 K 线特征 (诊断是否全为 0)
        kline_bars = kline_features.get("kline_bars_available", 0)
        if kline_bars > 0:
            print(f"   📊 K线特征: {kline_bars} bars, slope={kline_features.get('kline_trend_slope', 0):.4f}, vol={kline_features.get('kline_volatility', 0):.4f}")
        else:
            print(f"   ⚠️ 无 K 线数据 (特征全为 0)")

        return features

    def process_signal(self, signal: dict) -> dict:
        """处理单个信号, 做影子决策"""
        token_ca = signal["token_ca"]
        symbol = signal.get("symbol", "?")

        print(f"\n🔍 [{symbol}] 处理信号 #{signal['id']}...")

        # Step 1: 提取特征
        features = self._extract_features_for_signal(signal)

        # Step 2: GBDT 预测
        gbdt_output = self.gbdt.predict(features)
        print(f"   GBDT: p={gbdt_output['win_probability']:.3f} ({gbdt_output['confidence']})")

        # Step 3: Kronos 预测
        kronos_output = self.kronos.predict_for_token(token_ca)
        if kronos_output["pred_df"] is not None:
            print(f"   Kronos: trend={kronos_output['trend_direction']:+.4f}, "
                  f"vol={kronos_output['implied_volatility']:.4f}, "
                  f"conf={kronos_output['confidence']:.2f}")
        else:
            print(f"   Kronos: 降级 (无预测)")

        # Step 4: Kelly 计算
        kelly_output = self.kelly.calculate(gbdt_output, kronos_output)
        print(f"   Kelly: {kelly_output['decision']} — {kelly_output['reasoning']}")

        # Step 5: 记录到影子数据库
        self._save_decision(signal, gbdt_output, kronos_output, kelly_output)

        return kelly_output

    def _save_decision(self, signal: dict, gbdt: dict, kronos: dict, kelly: dict):
        """保存决策到影子数据库"""
        conn = sqlite3.connect(SHADOW_DB_PATH)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO shadow_decisions (
                    signal_id, token_ca, symbol, signal_ts,
                    gbdt_win_prob, gbdt_expected_pnl, gbdt_confidence,
                    kronos_trend_direction, kronos_trend_magnitude,
                    kronos_implied_volatility, kronos_confidence,
                    kronos_upside, kronos_downside,
                    kelly_decision, kelly_position_sol, kelly_win_prob,
                    kelly_odds, kelly_fraction, kelly_reasoning,
                    signal_market_cap, signal_holders, signal_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal["id"], signal["token_ca"], signal.get("symbol"),
                signal["timestamp"],
                gbdt["win_probability"], gbdt["expected_pnl"], gbdt["confidence"],
                kronos["trend_direction"], kronos["trend_magnitude"],
                kronos["implied_volatility"], kronos["confidence"],
                kronos["upside"], kronos["downside"],
                kelly["decision"], kelly["position_size_sol"],
                kelly["win_probability"], kelly["odds_ratio"],
                kelly["kelly_fraction"], kelly["reasoning"],
                signal.get("market_cap"), signal.get("holders"),
                signal.get("signal_type"),
            ))
            conn.commit()
        except Exception as e:
            print(f"⚠️  保存决策失败: {e}")
        finally:
            conn.close()

    def run(self):
        """主循环"""
        data_source = f"API: {SENTINEL_API_URL}" if self._api_client else f"本地: {SENTIMENT_DB_PATH}"

        print("=" * 60)
        print("🌙 Kronos Shadow Trader — 影子交易模式")
        print("=" * 60)
        print(f"   数据源: {data_source}")
        print(f"   影子DB: {SHADOW_DB_PATH}")
        print(f"   轮询: 每 {POLL_INTERVAL} 秒")
        print(f"   ⚠️  纯影子模式 — 不执行任何真实交易")
        print("=" * 60)

        # 初始化
        self.init_shadow_db()
        self._last_signal_id = self._get_last_processed_signal_id()
        print(f"   从信号 #{self._last_signal_id + 1} 开始")

        # 加载模型
        self.gbdt.load()
        # Kronos 延迟加载 (首次预测时)

        self.running = True

        # 优雅退出
        def handle_exit(signum, frame):
            print(f"\n🛑 收到退出信号, 保存状态...")
            self.running = False
        signal_module.signal(signal_module.SIGINT, handle_exit)
        signal_module.signal(signal_module.SIGTERM, handle_exit)

        # 主循环
        decisions_count = {"ENTER": 0, "SKIP": 0, "WAIT": 0}

        while self.running:
            try:
                signals = self._get_new_signals()

                if signals:
                    print(f"\n📥 发现 {len(signals)} 个新信号")
                    for signal in signals:
                        result = self.process_signal(signal)
                        decisions_count[result["decision"]] = decisions_count.get(result["decision"], 0) + 1
                        self._last_signal_id = signal["id"]
                        self._save_last_processed_signal_id(self._last_signal_id)

                    # 统计
                    total = sum(decisions_count.values())
                    print(f"\n📊 累计决策: {total} 总 | "
                          f"✅ ENTER {decisions_count['ENTER']} | "
                          f"❌ SKIP {decisions_count['SKIP']} | "
                          f"⏳ WAIT {decisions_count['WAIT']}")

                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 主循环异常: {e}")
                time.sleep(POLL_INTERVAL)

        print(f"\n🏁 影子交易已停止。最终统计:")
        print(f"   决策: {decisions_count}")
        print(f"   最后信号: #{self._last_signal_id}")


if __name__ == "__main__":
    trader = ShadowTrader()
    trader.run()
