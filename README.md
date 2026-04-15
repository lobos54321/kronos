# Kronos Shadow Trader

情绪驱动的量化预测影子交易系统。

## 架构

```
信号源 (只读现有 premium_signals DB)
    ↓
GBDT 特征筛选 → 入场概率 p_gbdt
    ↓
Kronos K线预测 → 趋势方向 + 波动率
    ↓
增强 Kelly 仓位 → 影子交易记录
    ↓
shadow_trades.db (独立数据库, 不影响现有系统)
```

## 与现有系统的关系

- **只读** 现有系统的 `premium_signals` 和 `kline_cache.db`
- **零修改** 现有系统的任何文件
- **独立数据库** `data/shadow_trades.db`
- **独立部署** 可以单独启停

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置数据源路径
cp .env.example .env
# 编辑 .env 指向现有系统的数据库路径

# 3. 训练 GBDT 模型 (首次)
python -m src.gbdt.train

# 4. 启动影子交易
python -m src.shadow_runner
```

## 目录结构

```
kronos-shadow/
├── src/
│   ├── gbdt/                  # GBDT 特征筛选引擎
│   │   ├── feature_extractor.py   # 从现有 DB 提取特征
│   │   ├── train.py               # 训练 LightGBM 模型
│   │   └── predictor.py           # 实时推理
│   ├── kronos/                # Kronos K线预测引擎
│   │   └── predictor.py           # Kronos 推理封装
│   ├── kelly/                 # 增强 Kelly 仓位引擎
│   │   └── enhanced_kelly.py      # GBDT + Kronos → Kelly
│   ├── shadow_runner.py       # 影子交易主循环
│   └── config.py              # 配置管理
├── data/                      # 本地数据 (gitignore)
├── models/                    # 训练好的模型 (gitignore)
├── requirements.txt
├── .env.example
└── .gitignore
```
