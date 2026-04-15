"""
GBDT 训练 — sklearn 版本 (不需要 libomp)

用于在不支持 LightGBM 原生库的环境中训练。
输出与 LightGBM 版本相同格式的模型文件。
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURE_COLUMNS, GBDT_MODEL_PATH, GBDT_MIN_TRAINING_SAMPLES
from src.gbdt.feature_extractor import extract_training_data


def main():
    print("=" * 60)
    print("🚀 Kronos Shadow — GBDT 训练 (sklearn 版)")
    print("=" * 60)

    # Step 1: 提取训练数据
    df = extract_training_data()

    if len(df) < GBDT_MIN_TRAINING_SAMPLES:
        print(f"\n❌ 训练数据不足: {len(df)} < {GBDT_MIN_TRAINING_SAMPLES}")
        return

    # Step 2: 确定可用特征
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    print(f"\n📋 可用特征: {len(available_features)}/{len(FEATURE_COLUMNS)}")

    X = df[available_features].copy().fillna(0)
    y_cls = df["is_winner"].values
    y_reg = np.clip(df["pnl_pct"].values, -1.0, 5.0)

    print(f"   样本数: {len(X)}, 正例: {y_cls.sum()} ({y_cls.mean()*100:.1f}%)")

    # ============================================================
    # Step 3: 训练分类器
    # ============================================================
    print(f"\n🔧 训练分类模型 (is_winner)")
    tscv = TimeSeriesSplit(n_splits=3)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_cls[train_idx], y_cls[val_idx]

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_prob) if len(set(y_val)) > 1 else 0.5
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_metrics.append({
            "fold": fold, "auc": auc, "accuracy": acc,
            "precision": prec, "recall": recall, "f1": f1,
        })
        print(f"   Fold {fold}: AUC={auc:.4f}  Acc={acc:.3f}  P={prec:.3f}  R={recall:.3f}  F1={f1:.3f}")

    # 全量训练最终分类器
    print(f"\n   全量 {len(X)} 条数据上训练最终分类器...")
    final_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8,
        random_state=42,
    )
    final_clf.fit(X, y_cls)

    # 特征重要性
    importances = final_clf.feature_importances_
    fi = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)

    print(f"\n📊 特征重要性排名:")
    for rank, (feat, imp) in enumerate(fi, 1):
        bar = "█" * int(imp / max(importances) * 30)
        print(f"   {rank:2d}. {feat:30s} {imp:.4f}  {bar}")

    # ============================================================
    # Step 4: 训练回归器
    # ============================================================
    print(f"\n🔧 训练回归模型 (pnl_pct)")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_reg[train_idx], y_reg[val_idx]

        reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"   Fold {fold}: RMSE={rmse:.4f}")

    final_reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8,
        random_state=42,
    )
    final_reg.fit(X, y_reg)

    # ============================================================
    # Step 5: 保存
    # ============================================================
    model_dir = Path(GBDT_MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "classifier": final_clf,
        "regressor": final_reg,
        "feature_columns": available_features,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_samples": len(X),
            "positive_rate": float(y_cls.mean()),
            "fold_metrics": fold_metrics,
            "feature_importance": [(f, float(i)) for f, i in fi],
            "engine": "sklearn_gbdt",
        }
    }

    with open(GBDT_MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n💾 模型已保存到 {GBDT_MODEL_PATH}")

    report_path = model_dir / "training_report.json"
    report = {
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(X),
        "positive_rate": float(y_cls.mean()),
        "fold_metrics": fold_metrics,
        "feature_importance": [(f, float(i)) for f, i in fi],
        "engine": "sklearn_gbdt",
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 训练报告: {report_path}")
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
