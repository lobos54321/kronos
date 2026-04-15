"""
影子交易 vs 现有系统对比分析

读取两个系统的交易记录, 做并排对比。
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.config import SHADOW_DB_PATH, PAPER_DB_PATH


def load_shadow_decisions() -> pd.DataFrame:
    """加载影子系统的所有决策"""
    conn = sqlite3.connect(SHADOW_DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM shadow_decisions
        ORDER BY signal_ts ASC
    """, conn)
    conn.close()
    return df


def load_existing_trades() -> pd.DataFrame:
    """加载现有系统的交易记录"""
    try:
        uri = f"file:{PAPER_DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        df = pd.read_sql_query("""
            SELECT id, token_ca, symbol, signal_ts, entry_price, entry_ts,
                   exit_price, exit_ts, exit_reason, pnl_pct, peak_pnl,
                   bars_held, signal_type, strategy_outcome
            FROM paper_trades
            WHERE pnl_pct IS NOT NULL
            ORDER BY entry_ts ASC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"⚠️  无法读取现有系统数据: {e}")
        return pd.DataFrame()


def compare_systems():
    """对比两套系统的决策和表现"""
    shadow = load_shadow_decisions()
    existing = load_existing_trades()

    print("=" * 70)
    print("📊 影子系统 vs 现有系统 — 对比报告")
    print("=" * 70)

    print(f"\n📋 数据量:")
    print(f"   影子系统决策: {len(shadow)} 条")
    print(f"   现有系统交易: {len(existing)} 条")

    if len(shadow) == 0:
        print("\n⚠️  影子系统暂无决策数据。请先运行 shadow_runner.py 积累数据。")
        return

    # --- 影子系统统计 ---
    print(f"\n🌙 影子系统决策分布:")
    if "kelly_decision" in shadow.columns:
        decision_dist = shadow["kelly_decision"].value_counts()
        for decision, count in decision_dist.items():
            print(f"   {decision}: {count} ({count/len(shadow)*100:.1f}%)")

    enters = shadow[shadow["kelly_decision"] == "ENTER"]
    skips = shadow[shadow["kelly_decision"] == "SKIP"]
    print(f"\n   ENTER 信号的平均 GBDT 胜率: {enters['gbdt_win_prob'].mean()*100:.1f}%")
    print(f"   SKIP 信号的平均 GBDT 胜率:  {skips['gbdt_win_prob'].mean()*100:.1f}%")

    # --- 如果有实际价格跟踪 ---
    if "actual_pnl_60min" in shadow.columns and shadow["actual_pnl_60min"].notna().any():
        tracked = shadow[shadow["actual_pnl_60min"].notna()]
        if len(tracked) > 0:
            print(f"\n📈 影子决策的实际表现 (60min 后):")

            enter_tracked = tracked[tracked["kelly_decision"] == "ENTER"]
            skip_tracked = tracked[tracked["kelly_decision"] == "SKIP"]

            if len(enter_tracked) > 0:
                enter_wr = (enter_tracked["actual_pnl_60min"] > 0).mean()
                enter_avg = enter_tracked["actual_pnl_60min"].mean()
                print(f"   ENTER 的 60min 胜率: {enter_wr*100:.1f}%, 平均 PnL: {enter_avg*100:.2f}%")

            if len(skip_tracked) > 0:
                skip_wr = (skip_tracked["actual_pnl_60min"] > 0).mean()
                skip_avg = skip_tracked["actual_pnl_60min"].mean()
                print(f"   SKIP  的 60min 胜率: {skip_wr*100:.1f}%, 平均 PnL: {skip_avg*100:.2f}%")

                if len(enter_tracked) > 0:
                    alpha = enter_avg - skip_avg
                    print(f"\n   🎯 Alpha (ENTER - SKIP): {alpha*100:.2f}%")
                    if alpha > 0:
                        print(f"   ✅ 影子系统的筛选有正向区分力!")
                    else:
                        print(f"   ❌ 影子系统的筛选暂无正向区分力")

    # --- 与现有系统交叉对比 ---
    if len(existing) > 0 and len(shadow) > 0:
        # 找到两个系统都见过的 token
        shadow_tokens = set(shadow["token_ca"].unique())
        existing_tokens = set(existing["token_ca"].unique())
        overlap = shadow_tokens & existing_tokens

        print(f"\n🔗 交叉对比:")
        print(f"   影子系统看过的 token: {len(shadow_tokens)}")
        print(f"   现有系统交易的 token: {len(existing_tokens)}")
        print(f"   重叠 token: {len(overlap)}")

        if len(overlap) > 0:
            # 对于重叠 token, 对比决策
            for token in list(overlap)[:10]:  # 展示前 10 个
                s_row = shadow[shadow["token_ca"] == token].iloc[0]
                e_rows = existing[existing["token_ca"] == token]
                e_pnl = e_rows["pnl_pct"].mean()

                print(f"\n   [{s_row.get('symbol', '?')}]")
                print(f"     影子决策: {s_row['kelly_decision']} (p={s_row['gbdt_win_prob']:.3f})")
                print(f"     现有结果: PnL={e_pnl*100:.2f}% ({len(e_rows)} 笔)")

    # --- 现有系统统计 (对照组) ---
    if len(existing) > 0:
        print(f"\n🏠 现有系统统计:")
        wr = (existing["pnl_pct"] > 0).mean()
        avg_pnl = existing["pnl_pct"].mean()
        print(f"   胜率: {wr*100:.1f}%")
        print(f"   平均 PnL: {avg_pnl*100:.2f}%")
        print(f"   中位 PnL: {existing['pnl_pct'].median()*100:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    compare_systems()
