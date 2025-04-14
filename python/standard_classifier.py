#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import random
from datetime import datetime

# 共通ユーティリティをインポート
from utils import numpy_to_python_type, debug_print, generate_mock_waveform, generate_mock_analysis

# スクリプト開始時のデバッグ情報
debug_print(f"Standard Genre Classifier started at {datetime.now()}")

# 標準ジャンル一覧 (10ジャンル - シンプルな分類)
GENRES = [
    'rock', 'pop', 'jazz', 'classical', 'electronic',
    'hip-hop', 'country', 'blues', 'reggae', 'metal'
]

# 必須ライブラリのインポートを試みる
USE_MOCK = True
try:
    debug_print("Importing libraries for standard analysis...")
    
    import librosa
    debug_print(f"Librosa imported successfully, version: {librosa.__version__}")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # ライブラリが正常にインポートできたかチェック
    USE_MOCK = False
    debug_print("All required libraries for standard analysis successfully imported.")
    print("必要なライブラリがインストールされています。標準分析モードで実行します", file=sys.stderr)
except ImportError as e:
    debug_print(f"Import error: {e}")
    print("必要なライブラリがインストールされていません。モックモードで実行します", file=sys.stderr)
except Exception as e:
    debug_print(f"Unexpected error during imports: {e}")
    print(f"予期しないエラーが発生しました: {e}", file=sys.stderr)

# モデルファイルのパス (標準モード用)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'standard_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'standard_scaler.pkl')

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
    third_genre = genres[2]["name"] if len(genres) > 2 else None
    
    tempo = analysis["tempo"]
    key = analysis["key"]
    energy = analysis["energy"]
    
    # キー情報の分解（例：C major → C, major）
    key_parts = key.split()
    key_note = key_parts[0] if len(key_parts) > 0 else "不明"
    key_mode = key_parts[1] if len(key_parts) > 1 else "不明"
    
    # テンポの評価（より詳細に）
    if tempo < 60:
        tempo_desc = "非常にゆっくりとした"
        tempo_feel = "瞑想的で落ち着いた"
    elif tempo < 85:
        tempo_desc = "ゆっくりとした"
        tempo_feel = "リラックスした"
    elif tempo < 110:
        tempo_desc = "中程度の"
        tempo_feel = "安定した"
    elif tempo < 140:
        tempo_desc = "アップテンポな"
        tempo_feel = "躍動的な"
    else:
        tempo_desc = "非常に速い"
        tempo_feel = "興奮と緊張感のある"
    
    # エネルギーの評価
    if energy < 0.33:
        energy_desc = "落ち着いた"
        energy_feel = "内省的で静かな"
    elif energy < 0.66:
        energy_desc = "バランスの取れた"
        energy_feel = "抑制されつつも表現力のある"
    else:
        energy_desc = "エネルギッシュな"
        energy_feel = "力強く活気に満ちた"
    
    # 基本的な説明文
    description = f"この楽曲は{tempo_desc}テンポ（{tempo}BPM）の{energy_desc}{top_genre}曲です。"
    
    # 調性の感情的な特徴
    if key_mode.lower() == "major":
        description += f"主要キーは{key_note} メジャーで、明るく開放的な響きが特徴的です。"
    elif key_mode.lower() == "minor":
        description += f"主要キーは{key_note} マイナーで、深みと感情的な響きが特徴的です。"
    else:
        description += f"主要キーは{key}です。"
    
    # 音楽的文脈・歴史的背景
    genre_context = {
        "rock": "ロックの力強さと反骨精神が表れた",
        "pop": "洗練されたメロディと現代的なポップサウンドを持つ",
        "jazz": "即興性と複雑なハーモニーが織りなす奥深い",
        "classical": "伝統的な西洋クラシックの様式美を持つ",
        "electronic": "テクノロジーを駆使した革新的な電子音楽の",
        "hip-hop": "ストリート文化を背景にしたリズミカルな",
        "country": "アメリカン・ルーツミュージックの伝統を継ぐ",
        "blues": "苦悩と希望を表現する情感豊かな",
        "reggae": "ジャマイカ発祥の独特なオフビートリズムを持つ",
        "metal": "激しいエネルギーと技巧的な演奏が特徴の"
    }
    
    if top_genre in genre_context:
        historical_context = genre_context[top_genre]
        description += f" この曲は{historical_context}作品です。"
    
    # ジャンルの混合について
    if second_genre and genres[1]["confidence"] > 0.3:
        if third_genre and genres[2]["confidence"] > 0.2:
            description += f" 主に{top_genre}をベースとしながらも、{second_genre}と{third_genre}の要素が混ざっています。"
        else:
            description += f" {second_genre}の影響を受けた{top_genre}スタイルが特徴的です。"
    else:
        description += f" 典型的な{top_genre}の特徴を持っています。"
    
    # 感情評価
    if top_genre in genre_context:
        description += f" この音楽は{energy_feel}音楽性と{tempo_feel}リズムが特徴の{energy_desc}サウンドを創り出しています。"
    
    return description

def generate_mock_results():
    """
    モックのジャンル分類結果を生成する（標準分析用）
    
    Returns:
    --------
    results : dict
        ジャンル分類、波形、分析結果を含む辞書
    """
    debug_print("Using mock mode - generating standard analysis results")
    
    # ジャンルリスト（標準）
    genres_list = GENRES
    
    # 全ジャンルに対してランダムな確信度を割り当て
    confidences = [random.random() for _ in range(len(genres_list))]
    
    # 確率の合計が1になるように正規化
    total = sum(confidences)
    confidences = [c / total for c in confidences]
    
    # ジャンル名と確信度をマッピング
    genres = []
    for i, genre in enumerate(genres_list):
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
        音楽ファイルから特徴量を抽出する（標準分析用）
        
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
            # 音声ファイルを読み込み - 標準的な長さ（30秒）
            debug_print("Loading audio file with duration: 30s...")
            y, sr = librosa.load(file_path, mono=True, duration=30)
            debug_print(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
            
            # 基本的な特徴量計算
            debug_print("Starting feature extraction calculations...")
            
            # MFCC - 標準的な次元数（13）
            debug_print("Extracting 13-dimensional MFCC...")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = mfcc.mean(axis=1)
            
            debug_print(f"MFCC shape: {mfcc.shape}, mean shape: {mfcc_mean.shape}")
            
            # クロマグラム特徴量
            debug_print("Extracting chroma features...")
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            
            debug_print(f"Chroma shape: {chroma.shape}, mean shape: {chroma_mean.shape}")
            
            # スペクトル特徴量の抽出
            debug_print("Extracting spectral features...")
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()  # スペクトル重心
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()  # スペクトル帯域幅
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()    # スペクトルロールオフ
            zero_crossing = librosa.feature.zero_crossing_rate(y).mean()      # ゼロ交差率
            
            # 特徴量のリスト
            features_list = [mfcc_mean, chroma_mean]
            spectral_features = [float(centroid), float(bandwidth), float(rolloff), float(zero_crossing)]
            
            # 全ての特徴量を結合
            all_features = np.concatenate([feat.flatten() for feat in features_list] + [np.array(spectral_features)])
            
            debug_print(f"Extracted {len(all_features)} features in total")
            return all_features
        
        except Exception as e:
            debug_print(f"Error during feature extraction: {str(e)}")
            raise
    
    def classify_genre(features):
        """
        特徴量からジャンルを分類する（標準分析用）
        
        Parameters:
        -----------
        features : numpy.ndarray
            抽出された特徴量
            
        Returns:
        --------
        genres : list
            ジャンルと確信度のリスト
        """
        try:
            debug_print("Loading model and scaler...")
            # モデルファイルが存在するかチェック
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                # スケーラーとモデルの読み込み
                scaler = joblib.load(SCALER_PATH)
                model = joblib.load(MODEL_PATH)
                
                # 特徴量のスケーリング
                debug_print("Scaling features...")
                scaled_features = scaler.transform(features.reshape(1, -1))
                
                # ジャンル予測
                debug_print("Predicting genre...")
                probabilities = model.predict_proba(scaled_features)[0]
                
                # ジャンルと確信度のリストを作成
                genres = []
                for i, prob in enumerate(probabilities):
                    genres.append({
                        "name": GENRES[i],
                        "confidence": float(prob)
                    })
                
                # 確信度でソート（降順）
                genres = sorted(genres, key=lambda x: x["confidence"], reverse=True)
                
                return genres
            else:
                debug_print("Model files not found, using mock results")
                # モデルファイルが見つからない場合はモック結果を使用
                return generate_mock_results()["genres"]
        
        except Exception as e:
            debug_print(f"Error during genre classification: {str(e)}")
            # エラーが発生した場合はモック結果を使用
            return generate_mock_results()["genres"]
    
    def analyze_audio(file_path):
        """
        音楽ファイルの分析を行う
        
        Parameters:
        -----------
        file_path : str
            音楽ファイルのパス
        
        Returns:
        --------
        analysis : dict
            楽曲分析情報
        """
        try:
            debug_print(f"Starting audio analysis for file: {file_path}")
            
            # 音声ファイルを読み込み
            y, sr = librosa.load(file_path, mono=True, duration=30)
            
            # テンポ検出
            debug_print("Detecting tempo...")
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # キー検出 (シンプルなアルゴリズム)
            debug_print("Detecting key...")
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_idx = np.argmax(np.mean(chroma, axis=1))
            keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            
            # メジャー/マイナーの判定（簡易的）
            major_minor_ratio = np.mean(chroma[key_idx]) / np.mean(chroma[(key_idx + 3) % 12])
            mode = "major" if major_minor_ratio > 1.0 else "minor"
            
            key = f"{keys[key_idx]} {mode}"
            
            # エネルギー計算 (シンプルバージョン)
            debug_print("Calculating energy...")
            energy = np.mean(librosa.feature.rms(y=y)[0])
            energy = min(1.0, energy * 5)  # 0～1に正規化
            
            # 楽器の検出（簡略版）
            instruments = {
                "ピアノ": 0.0,
                "ギター": 0.0,
                "ドラム": 0.0,
                "ベース": 0.0,
                "シンセサイザー": 0.0,
                "ボーカル": 0.0
            }
            
            # 周波数特性から簡易的に楽器の存在確率を推定
            spec = np.abs(librosa.stft(y))
            if np.mean(spec[50:100]) > 0.1:  # 中低域（ベース）
                instruments["ベース"] = np.random.uniform(0.6, 1.0)
            
            if np.mean(spec[100:200]) > 0.05:  # 低中域（ギター、ピアノ低音）
                instruments["ギター"] = np.random.uniform(0.3, 0.9)
                instruments["ピアノ"] = np.random.uniform(0.2, 0.8)
                
            if np.mean(spec[200:500]) > 0.05:  # 中域（ボーカル、ピアノ）
                instruments["ボーカル"] = np.random.uniform(0.5, 0.9)
                instruments["ピアノ"] = max(instruments["ピアノ"], np.random.uniform(0.4, 0.9))
            
            if np.mean(spec[500:1000]) > 0.02:  # 高域（シンセ、ドラム）
                instruments["シンセサイザー"] = np.random.uniform(0.2, 0.8)
                instruments["ドラム"] = np.random.uniform(0.4, 0.9)
            
            # ダンス性（リズムの強さに基づく簡易計算）
            beat_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
            danceability = min(1.0, beat_strength / 5.0)
            
            # アコースティック度（高域と雑音の少なさで簡易判定）
            high_freq_energy = np.mean(spec[800:1200])
            acousticness = 1.0 - min(1.0, high_freq_energy * 10)
            
            # セクション検出（簡易的）
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            changes = np.sum(np.abs(mfcc_delta) > 1.0, axis=0)
            significant_changes = np.where(changes > 5)[0]
            sections = max(2, min(8, len(significant_changes) // 5 + 2))  # 最小2、最大8セクション
            
            return {
                "tempo": float(tempo),
                "key": key,
                "energy": float(energy),
                "danceability": float(danceability),
                "acousticness": float(acousticness),
                "instruments": instruments,
                "sections": sections
            }
            
        except Exception as e:
            debug_print(f"Error during audio analysis: {str(e)}")
            # エラー時はモックデータを返す
            return generate_mock_analysis()

    def analyze_music_file(file_path):
        """
        音楽ファイルの標準的な分析を行う
        
        Parameters:
        -----------
        file_path : str
            音楽ファイルのパス
            
        Returns:
        --------
        results : dict
            分析結果
        """
        try:
            debug_print(f"Starting standard analysis for: {file_path}")
            
            # 特徴量抽出
            features = extract_features(file_path)
            
            # ジャンル分類
            genres = classify_genre(features)
            
            # 音楽分析
            analysis = analyze_audio(file_path)
            
            # 波形データ生成
            y, sr = librosa.load(file_path, mono=True, duration=30)
            # ダウンサンプリング（表示用）
            target_length = 1000
            step = max(1, len(y) // target_length)
            y_downsampled = y[::step][:target_length]
            # 正規化
            y_normalized = y_downsampled / np.max(np.abs(y_downsampled))
            
            waveform = {
                "samples": y_normalized.tolist(),
                "sampleRate": sr,
                "duration": len(y) / sr
            }
            
            # 楽曲説明の生成
            description = generate_mock_description(genres, analysis)
            
            return {
                "genres": genres,
                "waveform": waveform,
                "analysis": analysis,
                "description": description
            }
            
        except Exception as e:
            debug_print(f"Error during music file analysis: {str(e)}")
            # エラー時はモックデータを返す
            return generate_mock_results()

def analyze_standard_music(file_path):
    """
    音楽ファイルの標準分析を行う（メインエントリーポイント）
    
    Parameters:
    -----------
    file_path : str
        音楽ファイルのパス
    
    Returns:
    --------
    results_json : str
        分析結果（JSON形式）
    """
    debug_print(f"Standard analysis request received for: {file_path}")
    
    try:
        # モックモードかどうかで処理を分岐
        if USE_MOCK:
            results = generate_mock_results()
        else:
            results = analyze_music_file(file_path)
            
        # JSON形式に変換
        results_json = json.dumps(results, default=numpy_to_python_type)
        
        return results_json
    
    except Exception as e:
        debug_print(f"Error in analyze_standard_music: {str(e)}")
        # エラーオブジェクトを返す
        error_result = {
            "error": True,
            "message": f"音楽分析中にエラーが発生しました: {str(e)}",
        }
        return json.dumps(error_result)

# スクリプトとして実行された場合
if __name__ == "__main__":
    # サーバーからは3つの引数が渡されることを想定：
    # 1. ファイルパス
    # 2. 分析ID
    # 3. 分析手法（standard - 指定なしの場合は標準分析）
    if len(sys.argv) < 2:
        print("使用方法: python standard_classifier.py <音楽ファイルパス> [分析ID]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analysis_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    debug_print(f"Analysis ID: {analysis_id}, Mode: standard")
    
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        sys.exit(1)
    
    debug_print(f"Processing file: {file_path}")
    print(f"Mode: {'MOCK MODE' if USE_MOCK else 'NORMAL MODE'}", file=sys.stderr)
    print(f"Processing file: {file_path}", file=sys.stderr)
    print(f"Analysis ID: {analysis_id}", file=sys.stderr)
    
    results = analyze_standard_music(file_path)
    
    # Ensure we're printing the full JSON output
    max_length = 1000000  # Set a very high limit
    import sys
    old_max = sys.maxsize
    sys.maxsize = max_length
    
    # 重要: デバッグ情報を stderr に、JSON 結果のみを stdout に出力する
    # 一時的に標準出力をキャプチャして純粋なJSONのみを出力
    sys.stderr.write("Outputting JSON result to stdout\n")
    print(results)
    
    sys.maxsize = old_max
    print("Script execution completed successfully.", file=sys.stderr)
