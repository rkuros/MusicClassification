#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import time
import base64
import numpy as np
from datetime import datetime

# NumPy型をJSON互換の型に変換するヘルパー関数
def numpy_to_python_type(obj):
    """
    NumPy型をJSON互換のPython標準型に変換する
    
    Parameters:
    -----------
    obj : any
        変換する対象のオブジェクト
        
    Returns:
    --------
    Python標準型に変換されたオブジェクト
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # NumPy配列をリストに変換
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)  # NumPy整数型をPython intに変換
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)  # NumPy浮動小数点型をPython floatに変換
    elif isinstance(obj, (np.bool_)):
        return bool(obj)  # NumPy真偽型をPython boolに変換
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}  # 複素数を辞書に変換
    elif isinstance(obj, dict):
        return {k: numpy_to_python_type(v) for k, v in obj.items()}  # 辞書内の各値を再帰的に変換
    elif isinstance(obj, list):
        return [numpy_to_python_type(item) for item in obj]  # リスト内の各値を再帰的に変換
    else:
        return obj  # その他の型はそのまま返す

# デバッグ用ユーティリティ
def debug_print(message):
    """
    デバッグメッセージをログファイルと標準エラー出力に書き込む。
    標準出力には書き込まないことで、JSON解析に影響を与えないようにする。
    """
    try:
        with open("debug_log.txt", "a") as f:
            f.write(f"{message}\n")
        # 必ず標準エラー出力に出力し、標準出力には影響を与えないようにする
        print(f"DEBUG: {message}", file=sys.stderr)
        sys.stderr.flush()  # エラー出力をすぐにフラッシュして、出力の混在を避ける
    except Exception as e:
        print(f"デバッグログ書き込みエラー: {e}", file=sys.stderr)

# スクリプト開始時のデバッグ情報
debug_print(f"Script started at {datetime.now()}")

# ジャンル一覧
GENRES = [
    'rock', 'pop', 'jazz', 'classical', 'electronic',
    'hip-hop', 'country', 'blues', 'reggae', 'metal'
]

# 必須ライブラリのインポートを試みる
USE_MOCK = True
try:
    debug_print("Importing libraries...")
    import numpy as np
    debug_print(f"NumPy imported successfully, version: {np.__version__}")
    
    import librosa
    debug_print(f"Librosa imported successfully, version: {librosa.__version__}")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # ライブラリが正常にインポートできたかチェック
    USE_MOCK = False
    debug_print("All required libraries successfully imported.")
    print("必要なライブラリがインストールされています。本格的な分析モードで実行します", file=sys.stderr)
except ImportError as e:
    debug_print(f"Import error: {e}")
    print("必要なライブラリがインストールされていません。モックモードで実行します", file=sys.stderr)
except Exception as e:
    debug_print(f"Unexpected error during imports: {e}")
    print(f"予期しないエラーが発生しました: {e}", file=sys.stderr)

# モデルファイルのパス
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

def generate_mock_waveform():
    """
    モックの波形データを生成する
    
    Returns:
    --------
    waveform : dict
        波形データのダウンサンプリングされたリスト
    """
    # ランダムな波形データ生成（実際はオーディオ波形に似せた正弦波を使用）
    sample_count = 1000  # 表示用にダウンサンプリングされたサンプル数
    t = np.linspace(0, 10, sample_count)
    
    # 複数の正弦波を組み合わせて自然な波形に見せる
    wave_data = np.sin(2 * np.pi * t) * 0.5
    wave_data += np.sin(4 * np.pi * t) * 0.3
    wave_data += np.sin(8 * np.pi * t) * 0.2
    wave_data += np.random.normal(0, 0.1, sample_count)  # ノイズを追加
    
    # -1から1の間に正規化
    wave_data = wave_data / np.max(np.abs(wave_data))
    
    # base64エンコードされた文字列に変換
    wave_data_list = wave_data.tolist()
    
    return {
        "samples": wave_data_list,
        "sampleRate": 44100,
        "duration": 30.0  # 秒
    }

def generate_mock_analysis():
    """
    モックの楽曲分析結果を生成する
    
    Returns:
    --------
    analysis : dict
        楽曲分析情報
    """
    # ランダムなテンポ（BPM）
    tempo = random.randint(70, 180)
    
    # キー
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    modes = ["major", "minor"]
    key = random.choice(keys)
    mode = random.choice(modes)
    
    # 楽器構成
    instruments = ["ピアノ", "ギター", "ドラム", "ベース", "シンセサイザー", "ボーカル"]
    instrument_presence = {}
    for instrument in instruments:
        instrument_presence[instrument] = random.random()
    
    # エネルギー分布
    energy = random.uniform(0, 1)
    danceability = random.uniform(0, 1)
    acousticness = random.uniform(0, 1)
    
    # セクション数
    sections = random.randint(3, 8)
    
    return {
        "tempo": tempo,
        "key": f"{key} {mode}",
        "energy": energy,
        "danceability": danceability,
        "acousticness": acousticness,
        "instruments": instrument_presence,
        "sections": sections
    }

def generate_mock_description(genres, analysis):
    """
    モックの楽曲説明文を生成する
    
    Parameters:
    -----------
    genres : list
        ジャンルと確信度のリスト
    analysis : dict
        楽曲分析情報
        
    Returns:
    --------
    description : str
        楽曲の総合的な説明文
    """
    top_genre = genres[0]["name"]
    second_genre = genres[1]["name"] if len(genres) > 1 else None
    
    tempo = analysis["tempo"]
    key = analysis["key"]
    energy = analysis["energy"]
    
    # テンポの評価
    if tempo < 85:
        tempo_desc = "ゆっくりとした"
    elif tempo < 110:
        tempo_desc = "中程度の"
    else:
        tempo_desc = "アップテンポな"
    
    # エネルギーの評価
    if energy < 0.33:
        energy_desc = "落ち着いた"
    elif energy < 0.66:
        energy_desc = "バランスの取れた"
    else:
        energy_desc = "エネルギッシュな"
    
    # 基本的な説明文
    description = f"この楽曲は{tempo_desc}テンポ（{tempo}BPM）の{energy_desc}{top_genre}曲です。主要キーは{key}で、"
    
    # ジャンルの混合について
    if second_genre and genres[1]["confidence"] > 0.2:
        description += f"{second_genre}の要素も含まれています。"
    else:
        description += f"典型的な{top_genre}の特徴を持っています。"
    
    # 楽器構成について
    prominent_instruments = [k for k, v in analysis["instruments"].items() if v > 0.7]
    if prominent_instruments:
        description += f" 特に目立つ楽器は{', '.join(prominent_instruments)}です。"
    
    # セクション構造について
    description += f" 楽曲は約{analysis['sections']}つのセクションで構成されており、"
    
    # アコースティック度による追加説明
    if analysis["acousticness"] > 0.7:
        description += "アコースティックな雰囲気が強く、自然な音色が特徴的です。"
    elif analysis["acousticness"] < 0.3:
        description += "電子的な処理が多く施され、現代的なサウンドが特徴的です。"
    else:
        description += "アコースティックと電子的な要素がバランス良く混ざっています。"
    
    # ダンス性による追加説明
    if analysis["danceability"] > 0.7:
        description += " リズムが強調されたダンサブルな楽曲です。"
    
    return description

def generate_mock_results():
    """
    モックのジャンル分類結果を生成する（ライブラリがない場合のフォールバック）
    
    Returns:
    --------
    results : dict
        ジャンル分類、波形、分析結果を含む辞書
    """
    debug_print("Using mock mode - generating random results")
    
    # 全ジャンルに対してランダムな確信度を割り当て
    confidences = [random.random() for _ in range(len(GENRES))]
    
    # 確率の合計が1になるように正規化
    total = sum(confidences)
    confidences = [c / total for c in confidences]
    
    # ジャンル名と確信度をマッピング
    genres = []
    for i, genre in enumerate(GENRES):
        genres.append({
            "name": genre,
            "confidence": confidences[i]
        })
    
    # 確信度でソート（降順）
    genres = sorted(genres, key=lambda x: x["confidence"], reverse=True)
    genres = genres[:5]  # 上位5つの結果のみ
    
    # 波形データを生成
    waveform = generate_mock_waveform()
    
    # 楽曲分析を生成
    analysis = generate_mock_analysis()
    
    # 総合的な説明を生成
    description = generate_mock_description(genres, analysis)
    
    return {
        "genres": genres,
        "waveform": waveform,
        "analysis": analysis,
        "description": description
    }

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
        debug_print(f"Starting feature extraction for file: {file_path}")
        
        try:
            # 音声ファイルを読み込み
            debug_print("Loading audio file...")
            y, sr = librosa.load(file_path, mono=True, duration=30)
            debug_print(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
            
            # 基本的な特徴量計算
            debug_print("Starting feature extraction calculations...")
            
            debug_print("Extracting MFCC...")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)  # MFCC
            debug_print(f"MFCC shape: {mfcc.shape}, type: {type(mfcc)}")
            
            debug_print("Extracting chroma features...")
            chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)    # クロマグラム
            debug_print(f"Chroma shape: {chroma.shape}, type: {type(chroma)}")
            
            debug_print("Extracting spectral features...")
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()  # スペクトル重心
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  # スペクトル帯域幅
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()    # スペクトルロールオフ
            zero_crossing = librosa.feature.zero_crossing_rate(y).mean()      # ゼロ交差率
            
            # テンポ推定
            debug_print("Estimating tempo...")
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            debug_print(f"Estimated tempo: {tempo}")
            
            # RMS（音量の指標）
            debug_print("Calculating RMS energy...")
            rms = librosa.feature.rms(y=y).mean()
            
            debug_print(f"Python version: {sys.version}")
            debug_print(f"NumPy version: {np.__version__}")
            
            debug_print(f"Original mfcc shape: {mfcc.shape}")
            debug_print(f"Original chroma shape: {chroma.shape}")
            
            # 確実に1次元配列に変換
            mfcc = np.array(mfcc, dtype=float).flatten()
            chroma = np.array(chroma, dtype=float).flatten()
            
            # スカラー値を確実に浮動小数点数値に変換
            centroid = float(centroid)
            bandwidth = float(bandwidth)
            rolloff = float(rolloff)
            zero_crossing = float(zero_crossing)
            # tempo is an array [value], so extract the first element before converting to float
            tempo = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            rms = float(rms)
            
            # 個別の特徴量の値を表示
            debug_print(f"centroid: {centroid}")
            debug_print(f"bandwidth: {bandwidth}")
            debug_print(f"rolloff: {rolloff}")
            debug_print(f"zero_crossing: {zero_crossing}")
            debug_print(f"tempo: {tempo}")
            debug_print(f"rms: {rms}")
            
            other_features = np.array([centroid, bandwidth, rolloff, zero_crossing, tempo, rms], dtype=float)
            
            # デバッグ情報: 各配列の形状を確認
            debug_print(f"mfcc shape (after flatten): {mfcc.shape}")
            debug_print(f"chroma shape (after flatten): {chroma.shape}")
            debug_print(f"other_features shape: {other_features.shape}")
            debug_print(f"other_features data type: {other_features.dtype}")
            
            # すべての特徴量を1つの配列に結合
            try:
                features = np.concatenate([mfcc, chroma, other_features])
                debug_print(f"features shape after concatenate: {features.shape}")
                debug_print(f"features dtype: {features.dtype}")
            except Exception as e:
                debug_print(f"concatenate error: {str(e)}")
                debug_print(f"mfcc dtype: {mfcc.dtype}")
                debug_print(f"chroma dtype: {chroma.dtype}")
                debug_print(f"other_features dtype: {other_features.dtype}")
                raise
            
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

    def extract_waveform(y, sr):
        """
        オーディオ信号から波形データを抽出する
        
        Parameters:
        -----------
        y : numpy.ndarray
            オーディオ信号
        sr : int
            サンプリングレート
            
        Returns:
        --------
        waveform : dict
            波形データのダウンサンプリングされたリスト
        """
        debug_print("Extracting waveform data")
        try:
            # 表示用にダウンサンプリング（1000ポイント程度に）
            duration = len(y) / sr
            target_points = 1000
            
            # オーディオ長さに応じてダウンサンプリング率を調整
            if len(y) > target_points:
                factor = len(y) // target_points
                y_downsampled = y[::factor]
                if len(y_downsampled) > target_points:
                    y_downsampled = y_downsampled[:target_points]
            else:
                y_downsampled = y
                
            # 波形データをリスト化
            wave_data_list = y_downsampled.tolist()
            
            return {
                "samples": wave_data_list,
                "sampleRate": sr,
                "duration": duration
            }
        except Exception as e:
            debug_print(f"Error extracting waveform: {str(e)}")
            # エラーが発生した場合はダミー波形を返す
            return generate_mock_waveform()
    
    def analyze_audio(y, sr):
        """
        オーディオ信号から詳細な分析情報を抽出する
        
        Parameters:
        -----------
        y : numpy.ndarray
            オーディオ信号
        sr : int
            サンプリングレート
            
        Returns:
        --------
        analysis : dict
            詳細な音楽分析情報
        """
        debug_print("Analyzing audio in detail")
        try:
            # テンポ（BPM）推定
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
            
            # キー推定
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_indices = np.sum(chroma, axis=1)
            key_index = np.argmax(key_indices)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = keys[key_index]
            
            # メジャー/マイナー判定（簡易版）
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            # 循環シフトして相関を計算
            major_corrs = [np.corrcoef(np.roll(major_profile, i), key_indices)[0, 1] for i in range(12)]
            minor_corrs = [np.corrcoef(np.roll(minor_profile, i), key_indices)[0, 1] for i in range(12)]
            
            # 最も相関の高いモードを選択
            if max(major_corrs) > max(minor_corrs):
                mode = "major"
            else:
                mode = "minor"
            
            # エネルギー計算
            energy = librosa.feature.rms(y=y).mean()
            energy = float(energy) / 0.1  # 正規化（典型的なRMS値に基づく）
            energy = min(max(energy, 0.0), 1.0)  # 0〜1の範囲に収める
            
            # ダンス性（リズムの規則性に基づく）
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            danceability = np.mean(pulse) * 0.5  # スケール調整
            danceability = min(max(danceability, 0.0), 1.0)
            
            # アコースティック度（スペクトル特性に基づく）
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            acousticness = 1.0 - (np.mean(spectral_contrast[1:]) / 50.0)  # 高域のコントラストが低いほどアコースティック
            acousticness = min(max(acousticness, 0.0), 1.0)
            
            # セクション推定
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # MFCCの時間的変化を検出
            mfcc_delta = np.mean(np.abs(np.diff(mfccs, axis=1)), axis=0)
            
            # 変化率の大きい点をセクション境界と見なす
            threshold = np.mean(mfcc_delta) + np.std(mfcc_delta)
            boundaries = np.where(mfcc_delta > threshold)[0]
            
            # 近接した境界をマージ
            if len(boundaries) > 0:
                filtered_boundaries = [boundaries[0]]
                for boundary in boundaries[1:]:
                    if boundary - filtered_boundaries[-1] > sr * 5:  # 5秒以上離れていれば新しいセクション
                        filtered_boundaries.append(boundary)
                sections = len(filtered_boundaries) + 1
            else:
                sections = 1
                
            # 楽器推定（簡易版）
            instruments = {
                "ピアノ": 0.0,
                "ギター": 0.0,
                "ドラム": 0.0,
                "ベース": 0.0,
                "シンセサイザー": 0.0,
                "ボーカル": 0.0
            }
            
            # スペクトル特性から簡易的に楽器の存在確率を推定
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
            
            # ドラムの検出
            percussive = librosa.effects.percussive(y)
            drums_strength = librosa.feature.rms(y=percussive).mean() / librosa.feature.rms(y=y).mean()
            instruments["ドラム"] = min(max(drums_strength * 1.5, 0.0), 1.0)
            
            # ボーカルの検出
            harmonic = librosa.effects.harmonic(y)
            melspec = librosa.feature.melspectrogram(y=harmonic, sr=sr)
            vocal_range = np.mean(melspec[30:70, :]) / np.mean(melspec)  # ボーカル周波数帯の強さ
            instruments["ボーカル"] = min(max(vocal_range * 2.0 - 0.3, 0.0), 1.0)
            
            # ベースの検出
            bass_range = np.mean(melspec[:20, :]) / np.mean(melspec)
            instruments["ベース"] = min(max(bass_range * 2.0 - 0.2, 0.0), 1.0)
            
            # ギター/ピアノ/シンセの推定（簡易的）
            if spectral_flatness > 0.01:  # シンセの特徴
                instruments["シンセサイザー"] = min(spectral_flatness * 50, 1.0)
            
            if 1500 < spectral_centroid < 3000 and spectral_bandwidth < 2000:
                instruments["ピアノ"] = 0.7
                
            if 800 < spectral_centroid < 2500:
                instruments["ギター"] = 0.6
            
            return {
                "tempo": tempo,
                "key": f"{key} {mode}",
                "energy": energy,
                "danceability": danceability,
                "acousticness": acousticness,
                "instruments": instruments,
                "sections": sections
            }
            
        except Exception as e:
            debug_print(f"Error analyzing audio: {str(e)}")
            # エラーが発生した場合はダミーデータを返す
            return generate_mock_analysis()
    
    def generate_description(genres, analysis):
        """
        ジャンルと音楽分析に基づいて、楽曲の総合的な説明を生成する
        
        Parameters:
        -----------
        genres : list
            ジャンルと確信度のリスト
        analysis : dict
            楽曲分析情報
            
        Returns:
        --------
        description : str
            楽曲の総合的な説明文
        """
        try:
            top_genre = genres[0]["name"]
            second_genre = genres[1]["name"] if len(genres) > 1 else None
            
            tempo = analysis["tempo"]
            key = analysis["key"]
            energy = analysis["energy"]
            
            # テンポの評価
            if tempo < 85:
                tempo_desc = "ゆっくりとした"
            elif tempo < 110:
                tempo_desc = "中程度の"
            else:
                tempo_desc = "アップテンポな"
            
            # エネルギーの評価
            if energy < 0.33:
                energy_desc = "落ち着いた"
            elif energy < 0.66:
                energy_desc = "バランスの取れた"
            else:
                energy_desc = "エネルギッシュな"
            
            # 基本的な説明文
            description = f"この楽曲は{tempo_desc}テンポ（{tempo:.1f}BPM）の{energy_desc}{top_genre}曲です。主要キーは{key}で、"
            
            # ジャンルの混合について
            if second_genre and genres[1]["confidence"] > 0.2:
                description += f"{second_genre}の要素も含まれています。"
            else:
                description += f"典型的な{top_genre}の特徴を持っています。"
            
            # 楽器構成について
            prominent_instruments = [k for k, v in analysis["instruments"].items() if v > 0.7]
            if prominent_instruments:
                description += f" 特に目立つ楽器は{', '.join(prominent_instruments)}です。"
            
            # セクション構造について
            description += f" 楽曲は約{analysis['sections']}つのセクションで構成されており、"
            
            # アコースティック度による追加説明
            if analysis["acousticness"] > 0.7:
                description += "アコースティックな雰囲気が強く、自然な音色が特徴的です。"
            elif analysis["acousticness"] < 0.3:
                description += "電子的な処理が多く施され、現代的なサウンドが特徴的です。"
            else:
                description += "アコースティックと電子的な要素がバランス良く混ざっています。"
            
            # ダンス性による追加説明
            if analysis["danceability"] > 0.7:
                description += " リズムが強調されたダンサブルな楽曲です。"
                
            return description
        except Exception as e:
            debug_print(f"Error generating description: {str(e)}")
            return "分析情報からの詳細説明を生成できませんでした。"
    
    def predict_genre(file_path):
        """
        音楽ファイルのジャンルを予測し、波形データと詳細分析を行う

        Parameters:
        -----------
        file_path : str
            音楽ファイルのパス

        Returns:
        --------
        results : dict
            ジャンル分類、波形、分析結果を含む辞書
        """
        debug_print(f"Analyzing file: {file_path}")
        
        # 音声ファイルを読み込み
        try:
            y, sr = librosa.load(file_path, mono=True, duration=30)
            debug_print(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
        except Exception as e:
            debug_print(f"Error loading audio: {str(e)}")
            return generate_mock_results()  # エラー時はモックデータを返す
        
        # 特徴量抽出
        features = extract_features(file_path)
        
        # 波形データ抽出
        waveform = extract_waveform(y, sr)
        
        # 音楽分析
        analysis = analyze_audio(y, sr)
        
        # モデルとスケーラーのロード、または新規作成
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        else:
            debug_print("モデルが存在しないため、新規に作成します")
            clf, scaler = train_model()
            # モデルとスケーラーの保存
            joblib.dump(clf, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
        
        # 特徴量のスケーリングと予測
        debug_print(f"features before flatten: {features.shape}")
        features = features.flatten()  # 確実に1次元にする
        debug_print(f"features after flatten: {features.shape}")
        
        try:
            features_scaled = scaler.transform(features.reshape(1, -1))
            debug_print(f"features_scaled shape: {features_scaled.shape}")
        except Exception as e:
            debug_print(f"scaling error: {str(e)}")
            debug_print(f"features shape: {features.shape}")
            debug_print(f"features content: {features}")
            raise
        probabilities = clf.predict_proba(features_scaled)[0]
        
        # ジャンルと確信度のリストを作成
        genres = []
        for i, genre in enumerate(GENRES):
            genres.append({
                "name": genre,
                "confidence": float(probabilities[i])
            })
        
        # 確信度でソート（降順）
        genres = sorted(genres, key=lambda x: x["confidence"], reverse=True)
        
        # 上位5つの結果のみ取得
        genres = genres[:5]
        
        # 総合的な説明を生成
        description = generate_description(genres, analysis)
        
        # 結果を辞書にまとめる
        return {
            "genres": genres,
            "waveform": waveform,
            "analysis": analysis,
            "description": description
        }

def main():
    """
    メイン関数
    コマンドライン引数からMP3ファイルのパスと分析IDを取得し、
    ジャンル分類結果をJSON形式で標準出力に出力する
    """
    debug_print("Starting main function")
    
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
        debug_print("Brief pause for user experience")
        time.sleep(1.5)
        
        if USE_MOCK:
            # モックのジャンル分類結果を生成
            debug_print("Using mock mode")
            genres = generate_mock_results()
        else:
            # 本格的なジャンル予測の実行
            debug_print("Using full analysis mode")
            genres = predict_genre(audio_path)
        
        # 結果をJSON形式で出力
        result = {
            "analysisId": analysis_id,
            "genres": genres
        }
        
        # NumPy型をPython標準型に変換
        result = numpy_to_python_type(result)
        
        # バッファをフラッシュして、出力の混在を防ぐ
        sys.stderr.flush()
        
        # JSONを標準出力に送る (標準エラー出力ではない)
        # これがサーバーに解析される出力
        print(json.dumps(result))
        sys.stdout.flush()
        
    except Exception as e:
        print(json.dumps({"error": f"処理エラー: {str(e)}"}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    debug_print("Script execution started")
    try:
        main()
        debug_print("Script completed successfully")
    except Exception as e:
        debug_print(f"Uncaught exception: {e}")
        import traceback
        debug_print(traceback.format_exc())
        sys.exit(1)
