#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import time
from datetime import datetime

# ジャンル一覧
GENRES = [
    'rock', 'pop', 'jazz', 'classical', 'electronic',
    'hip-hop', 'country', 'blues', 'reggae', 'metal'
]

# 必須ライブラリのインポートを試みる
USE_MOCK = True
try:
    import numpy as np
    import librosa
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # ライブラリが正常にインポートできたかチェック
    USE_MOCK = False
    print("必要なライブラリがインストールされています。本格的な分析モードで実行します", file=sys.stderr)
except ImportError:
    print("必要なライブラリがインストールされていません。モックモードで実行します", file=sys.stderr)

# モデルファイルのパス
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

def generate_mock_results():
    """
    モックのジャンル分類結果を生成する（ライブラリがない場合のフォールバック）
    
    Returns:
    --------
    results : list
        ジャンルと確信度のリスト（ランダム値）
    """
    # 全ジャンルに対してランダムな確信度を割り当て
    confidences = [random.random() for _ in range(len(GENRES))]
    
    # 確率の合計が1になるように正規化
    total = sum(confidences)
    confidences = [c / total for c in confidences]
    
    # ジャンル名と確信度をマッピング
    results = []
    for i, genre in enumerate(GENRES):
        results.append({
            "name": genre,
            "confidence": confidences[i]
        })
    
    # 確信度でソート（降順）
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    # 上位5つの結果のみ返す
    return results[:5]

if not USE_MOCK:
    def extract_features(file_path):
        """
        音楽ファイルから特徴量を抽出する

        Parameters:
        -----------
        file_path : str
            音楽ファイルのパス

        Returns:
        --------
        features : numpy.ndarray
            抽出された特徴量
        """
        try:
            # 音声ファイルを読み込み
            y, sr = librosa.load(file_path, mono=True, duration=30)
            
            # 基本的な特徴量計算
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)  # MFCC
            chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)    # クロマグラム
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()  # スペクトル重心
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  # スペクトル帯域幅
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()    # スペクトルロールオフ
            zero_crossing = librosa.feature.zero_crossing_rate(y).mean()      # ゼロ交差率
            
            # テンポ推定
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # RMS（音量の指標）
            rms = librosa.feature.rms(y=y).mean()
            
            # すべての特徴量を1つの配列に結合
            features = np.concatenate([
                mfcc, 
                chroma, 
                [centroid, bandwidth, rolloff, zero_crossing, tempo, rms]
            ])
            
            return features
        
        except Exception as e:
            print(json.dumps({"error": f"特徴量抽出エラー: {str(e)}"}), file=sys.stderr)
            raise

    def train_model():
        """
        モックのモデルを作成する（テスト用）
        実際のシステムでは、事前に収集した音楽データで訓練するべき
        
        Returns:
        --------
        clf : RandomForestClassifier
            訓練済みモデル
        scaler : StandardScaler
            スケーラー
        """
        # 特徴量の次元数（今回の例では13 + 12 + 6 = 31次元）
        feature_dim = 31
        
        # 仮想的なトレーニングデータを生成
        # 実際にはジャンルごとのオーディオデータから特徴量を抽出すべき
        X_train = np.random.rand(100 * len(GENRES), feature_dim)
        y_train = np.repeat(np.arange(len(GENRES)), 100)
        
        # 特徴量のスケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # モデルの訓練
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        return clf, scaler

    def predict_genre(file_path):
        """
        音楽ファイルのジャンルを予測する

        Parameters:
        -----------
        file_path : str
            音楽ファイルのパス

        Returns:
        --------
        results : list
            ジャンルと確信度のリスト
        """
        # 特徴量抽出
        features = extract_features(file_path)
        
        # モデルとスケーラーのロード、または新規作成
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        else:
            print("モデルが存在しないため、新規に作成します", file=sys.stderr)
            clf, scaler = train_model()
            # モデルとスケーラーの保存
            joblib.dump(clf, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
        
        # 特徴量のスケーリングと予測
        features_scaled = scaler.transform(features.reshape(1, -1))
        probabilities = clf.predict_proba(features_scaled)[0]
        
        # ジャンルと確信度のリストを作成
        results = []
        for i, genre in enumerate(GENRES):
            results.append({
                "name": genre,
                "confidence": float(probabilities[i])
            })
        
        # 確信度でソート（降順）
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        
        # 上位5つの結果のみ返す
        return results[:5]

def main():
    """
    メイン関数
    コマンドライン引数からMP3ファイルのパスと分析IDを取得し、
    ジャンル分類結果をJSON形式で標準出力に出力する
    """
    # コマンドライン引数のチェック
    if len(sys.argv) < 3:
        print(json.dumps({"error": "引数が不足しています: MP3ファイルのパスと分析IDが必要です"}), file=sys.stderr)
        sys.exit(1)
    
    # 引数の取得
    audio_path = sys.argv[1]
    analysis_id = sys.argv[2]
    
    # ファイルの存在チェック
    if not os.path.exists(audio_path):
        print(json.dumps({"error": f"ファイルが見つかりません: {audio_path}"}), file=sys.stderr)
        sys.exit(1)
    
    try:
        # 処理中の演出として少し待機
        time.sleep(1.5)
        
        if USE_MOCK:
            # モックのジャンル分類結果を生成
            genres = generate_mock_results()
        else:
            # 本格的なジャンル予測の実行
            genres = predict_genre(audio_path)
        
        # 結果をJSON形式で出力
        result = {
            "analysisId": analysis_id,
            "genres": genres
        }
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": f"処理エラー: {str(e)}"}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
