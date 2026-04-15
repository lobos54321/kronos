"""
GBDT 训练引擎

用 LightGBM 训练二分类模型 (is_winner) 和回归模型 (pnl_pct)。
输出: 特征重要性排名 + 保存训练好的模型。
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import FEATURE_COLUMNS, GBDT_MODEL_PATH, GBDT_MIN_TRAINING_SAMPLES
from src.gbdt.feature_extractor import extract_training_data


def train_classifier(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    训练 LightGBM 分类模型，预测 is_winner (入场后盈利概率)。

    使用 TimeSeriesSplit 而非随机拆分，防止前瞻偏差。
    """
    # 过滤掉缺失特征过多的行
    X = df[feature_cols].copy()
    y = df["is_winner"].values

    # 填充 NaN
    X = X.fillna(0)

    print(f"\n🔧 训练分类模型 (is_winner)")
    print(f"   样本数: {len(X)}, 正例: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   特征数: {len(feature_cols)}")

    # 时间序列交叉验证 (3 折, 按时间顺序)
    tscv = TimeSeriesSplit(n_splits=3)
    fold_metrics = []

    best_model = None
    best_auc = -1

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # LightGBM 参数 (保守, 防过拟合)
        params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "num_leaves": 16,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "random_state": 42,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],  # 静默
        )

        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_prob) if len(set(y_val)) > 1 else 0.5
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_metrics.append({
            "fold": fold, "auc": auc, "accuracy": acc,
            "precision": prec, "recall": recall, "f1": f1,
            "val_size": len(y_val), "val_positive_rate": float(y_val.mean()),
        })

        print(f"   Fold {fold}: AUC={auc:.4f}  Acc={acc:.3f}  P={prec:.3f}  R={recall:.3f}  F1={f1:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # 在全量数据上重新训练最终模型
    print(f"\n   在全量 {len(X)} 条数据上训练最终模型...")
    final_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 16,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "random_state": 42,
    }
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y)

    # 特征重要性
    importances = final_model.feature_importances_
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    print(f"\n📊 特征重要性排名:")
    for rank, (feat, imp) in enumerate(fi, 1):
        bar = "█" * int(imp / max(importances) * 30)
        print(f"   {rank:2d}. {feat:30s} {imp:6.0f}  {bar}")

    return {
        "model": final_model,
        "feature_columns": feature_cols,
        "fold_metrics": fold_metrics,
        "feature_importance": fi,
        "training_samples": len(X),
        "positive_rate": float(y.mean()),
    }


def train_regressor(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    训练 LightGBM 回归模型，预测 pnl_pct。
    用于估算赔率 (odds ratio)。
    """
    X = df[feature_cols].copy().fillna(0)
    y = df["pnl_pct"].values

    # 裁剪极端值 (防止离群点主导训练)
    y_clipped = np.clip(y, -1.0, 5.0)

    print(f"\n🔧 训练回归模型 (pnl_pct)")
    print(f"   样本数: {len(X)}, median PnL: {np.median(y)*100:.2f}%")

    tscv = TimeSeriesSplit(n_splits=3)
    rmse_list = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_clipped[train_idx], y_clipped[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=16, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            verbosity=-1, random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],
        )

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse_list.append(rmse)
        print(f"   Fold {fold}: RMSE={rmse:.4f}")

    # 全量训练
    final_model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        num_leaves=16, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        verbosity=-1, random_state=42,
    )
    final_model.fit(X, y_clipped)

    return {
        "model": final_model,
        "fold_rmse": rmse_list,
    }


def main():
    print("=" * 60)
    print("🚀 Kronos Shadow — GBDT 训练流程")
    print("=" * 60)

    # Step 1: 提取训练数据
    df = extract_training_data()

    if len(df) < GBDT_MIN_TRAINING_SAMPLES:
        print(f"\n❌ 训练数据不足: {len(df)} < {GBDT_MIN_TRAINING_SAMPLES}")
        print("   需要更多已关闭的交易数据。现有系统继续运行积累数据即可。")
        return

    # Step 2: 确定可用特征
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    print(f"\n📋 可用特征: {len(available_features)}/{len(FEATURE_COLUMNS)}")

    # Step 3: 训练分类器
    clf_result = train_classifier(df, available_features)

    # Step 4: 训练回归器
    reg_result = train_regressor(df, available_features)

    # Step 5: 保存模型
    model_dir = Path(GBDT_MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "classifier": clf_result["model"],
        "regressor": reg_result["model"],
        "feature_columns": available_features,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "training_samples": clf_result["training_samples"],
            "positive_rate": clf_result["positive_rate"],
            "fold_metrics": clf_result["fold_metrics"],
            "feature_importance": clf_result["feature_importance"],
            "fold_rmse": reg_result["fold_rmse"],
        }
    }

    with open(GBDT_MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n💾 模型已保存到 {GBDT_MODEL_PATH}")

    # 保存分析报告
    report_path = model_dir / "training_report.json"
    report = {
        "trained_at": datetime.now().isoformat(),
        "training_samples": clf_result["training_samples"],
        "positive_rate": clf_result["positive_rate"],
        "fold_metrics": clf_result["fold_metrics"],
        "feature_importance": [(f, int(i)) for f, i in clf_result["feature_importance"]],
        "fold_rmse": [float(r) for r in reg_result["fold_rmse"]],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 训练报告已保存到 {report_path}")
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
