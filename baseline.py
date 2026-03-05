#!/usr/bin/env python3
"""
EV Charging Station Load Forecasting Baseline
使用 LightGBM 模型预测充电站负荷

预测目标: 2024/11/1 0:00 - 2024/12/31 23:45 (61天, 5856条数据)
输出文件: submission.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor  # Import GradientBoostingRegressor

# 尝试导入 LightGBM，如果不存在则使用 sklearn 的 GradientBoosting
USE_LIGHTGBM = False
try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
    print("使用 LightGBM 模型")
except Exception as e:
    print(f"LightGBM 不可用 ({type(e).__name__})，使用 sklearn GradientBoostingRegressor")


def load_training_data(filepath):
    """加载训练数据"""
    print(f"加载训练数据: {filepath}")
    # 跳过第一行中文列名，使用第二行英文列名
    df = pd.read_csv(filepath, encoding='gbk', skiprows=1)
    df['TIME'] = pd.to_datetime(df['TIME'])
    print(f"  数据形状: {df.shape}")
    print(f"  时间范围: {df['TIME'].min()} - {df['TIME'].max()}")
    return df


def create_features(df):
    """创建时间特征"""
    df = df.copy()
    df['hour'] = df['TIME'].dt.hour
    df['minute'] = df['TIME'].dt.minute
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15  # 0-95
    df['dayofweek'] = df['TIME'].dt.dayofweek
    df['month'] = df['TIME'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    return df


def train_model(train_df):
    """训练模型"""
    feature_cols = ['hour', 'minute', 'time_slot', 'dayofweek', 'month', 'is_weekend']

    X_train = train_df[feature_cols]
    y_train = train_df['V']

    print(f"\n训练模型...")
    print(f"  特征: {feature_cols}")
    print(f"  样本数: {len(X_train)}")

    if USE_LIGHTGBM:
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=64,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

    model.fit(X_train, y_train)
    print("  训练完成")

    # 训练集上的 RMSE
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    print(f"  训练集 RMSE: {train_rmse:.4f}")

    return model, feature_cols


def generate_prediction_times():
    """生成预测时间范围: 2024/11/1 - 2024/12/31"""
    start_date = datetime(2024, 11, 1, 0, 0)
    end_date = datetime(2024, 12, 31, 23, 45)

    times = []
    current = start_date
    while current <= end_date:
        times.append(current)
        current += timedelta(minutes=15)

    return times


def predict_and_save(model, feature_cols, output_path):
    """生成预测并保存"""
    print(f"\n生成预测...")

    # 生成预测时间
    pred_times = generate_prediction_times()
    print(f"  预测时间范围: {pred_times[0]} - {pred_times[-1]}")
    print(f"  预测样本数: {len(pred_times)}")

    # 创建预测 DataFrame
    pred_df = pd.DataFrame({'TIME': pred_times})
    pred_df = create_features(pred_df)

    # 预测
    X_pred = pred_df[feature_cols]
    predictions = model.predict(X_pred)

    # 确保预测值非负
    predictions = np.maximum(predictions, 0)

    # 格式化输出
    output_df = pd.DataFrame({
        'TIME': [t.strftime('%Y/%-m/%-d %-H:%M') for t in pred_times],
        'V': predictions
    })

    # 保存
    output_df.to_csv(output_path, index=False, encoding='gbk')
    print(f"  预测结果已保存: {output_path}")
    print(f"  输出行数: {len(output_df)}")

    # 预测统计
    print(f"\n预测统计:")
    print(f"  最小值: {predictions.min():.4f} MW")
    print(f"  最大值: {predictions.max():.4f} MW")
    print(f"  均值: {predictions.mean():.4f} MW")
    print(f"  标准差: {predictions.std():.4f} MW")

    return output_df


def main():
    print("=" * 50)
    print("EV Charging Station Load Forecasting Baseline")
    print("=" * 50)

    # 1. 加载训练数据
    train_df = load_training_data('A榜-充电站充电负荷训练数据.csv')

    # 2. 特征工程
    train_df = create_features(train_df)

    # 3. 训练模型
    model, feature_cols = train_model(train_df)

    # 4. 预测并保存
    output_df = predict_and_save(model, feature_cols, 'submission.csv')

    print("\n" + "=" * 50)
    print("完成! 请提交 submission.csv")
    print("=" * 50)


if __name__ == '__main__':
    main()
