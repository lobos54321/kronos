"""
GBDT 实时推理器

加载训练好的模型，对新信号做实时预测。
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import GBDT_MODEL_PATH


class GBDTPredictor:
    """GBDT 入场概率预测器"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or GBDT_MODEL_PATH
        self.classifier = None
        self.regressor = None
        self.feature_columns = None
        self.metadata = None
        self._loaded = False

    def load(self) -> bool:
        """加载训练好的模型"""
        path = Path(self.model_path)
        if not path.exists():
            print(f"⚠️  GBDT 模型未找到: {self.model_path}")
            print(f"   请先运行: python -m src.gbdt.train")
            return False

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        self.classifier = bundle["classifier"]
        self.regressor = bundle["regressor"]
        self.feature_columns = bundle["feature_columns"]
        self.metadata = bundle["metadata"]
        self._loaded = True

        print(f"✅ GBDT 模型已加载 (训练于 {self.metadata['trained_at']})")
        print(f"   训练样本: {self.metadata['training_samples']}, 正例率: {self.metadata['positive_rate']*100:.1f}%")
        return True

    def predict(self, features: dict) -> dict:
        """
        对单个信号做预测。

        Args:
            features: 从 feature_extractor 提取的特征字典

        Returns:
            {
                "win_probability": float,  # 盈利概率 (0-1)
                "expected_pnl": float,     # 预期 PnL (%)
                "confidence": str,         # HIGH / MEDIUM / LOW
                "top_features": list,      # 最重要的 3 个特征
            }
        """
        if not self._loaded:
            if not self.load():
                return {
                    "win_probability": 0.3,  # 降级到默认值
                    "expected_pnl": 0.0,
                    "confidence": "FALLBACK",
                    "top_features": [],
                }

        # 构建特征向量
        X = pd.DataFrame([{
            col: features.get(col, 0) for col in self.feature_columns
        }])
        X = X.fillna(0)

        # 分类预测: 盈利概率
        win_prob = float(self.classifier.predict_proba(X)[0, 1])

        # 回归预测: 预期 PnL
        expected_pnl = float(self.regressor.predict(X)[0])

        # 置信度分级
        if win_prob >= 0.5:
            confidence = "HIGH"
        elif win_prob >= 0.35:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # 特征贡献 (用 GBDT 的 feature importance 近似)
        fi = self.metadata.get("feature_importance", [])
        top_features = [f"{name} ({imp})" for name, imp in fi[:3]]

        return {
            "win_probability": win_prob,
            "expected_pnl": expected_pnl,
            "confidence": confidence,
            "top_features": top_features,
        }

    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        """批量预测"""
        return [self.predict(f) for f in features_list]
