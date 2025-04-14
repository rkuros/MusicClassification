#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import random

# 標準ジャンル一覧（standard_classifier.pyと同じ）
GENRES = [
    'rock', 'pop', 'jazz', 'classical', 'electronic',
    'hip-hop', 'country', 'blues', 'reggae', 'metal'
]

print("Creating standard models for music classification...")

# モデルファイルのパス
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'standard_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'standard_scaler.pkl')

# シンプルなデモ用のデータを作成
n_samples = 100  # 各ジャンル10サンプル
n_features = 29  # 標準分析の特徴量次元数（13 MFCC + 12 chroma + 4 spectral features）

# ランダムデータ生成
X = np.random.rand(n_samples, n_features)
y = np.repeat(np.arange(len(GENRES)), n_samples // len(GENRES))

# スケーラーの作成と適用
print("Creating and fitting standard scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ランダムフォレストモデルの作成（シンプルな設定）
print("Training standard RandomForest model...")
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_scaled, y)

# モデルとスケーラーの保存
print(f"Saving standard scaler to {SCALER_PATH}")
joblib.dump(scaler, SCALER_PATH)

print(f"Saving standard model to {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print("Standard models created successfully!")
