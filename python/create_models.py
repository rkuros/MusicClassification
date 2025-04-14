#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 共通のジャンル一覧（advanced_classifier.pyと同じ）
GENRES = [
    'rock', 'pop', 'jazz', 'classical', 'electronic',
    'hip-hop', 'country', 'blues', 'reggae', 'metal',
    'dance', 'funk', 'soul', 'folk', 'ambient', 'indie'
]

print("Creating simple models for music classification...")

# モデルファイルのパス
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'advanced_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'advanced_scaler.pkl')

# シンプルなデモ用のデータを作成
n_samples = 160  # 各ジャンル10サンプル
n_features = 60  # 音楽特徴量の次元数

# ランダムデータ生成（実際にはMFCC、クロマグラム、スペクトル特性などの特徴量）
X = np.random.rand(n_samples, n_features)
y = np.repeat(np.arange(len(GENRES)), n_samples // len(GENRES))

# スケーラーの作成と適用
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ランダムフォレストモデルの作成
model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
model.fit(X_scaled, y)

# モデルとスケーラーの保存
print(f"Saving scaler to {SCALER_PATH}")
joblib.dump(scaler, SCALER_PATH)

print(f"Saving model to {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print("Models created successfully!")
