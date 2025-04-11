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
    elif isinstance(obj, (np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)  # NumPy整数型をPython intに変換
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)  # NumPy浮動小数点型をPython floatに変換
    elif isinstance(obj, np.bool_):
        return bool(obj)  # NumPy真偽型をPython boolに変換
    elif isinstance(obj, (np.complex64, np.complex128)):
        return {'real': float(obj.real), 'imag': float(obj.imag)}  # 複素数を辞書に変換
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
    
    # ジャンルの混合について（より詳細に）
    if second_genre and genres[1]["confidence"] > 0.3:
        if third_genre and genres[2]["confidence"] > 0.2:
            description += f" 主に{top_genre}をベースとしながらも、{second_genre}と{third_genre}の要素が融合した多面的な音楽性を持っています。"
        else:
            description += f" {second_genre}の影響を強く受けた{top_genre}スタイルが特徴的です。"
    else:
        description += f" 典型的な{top_genre}の特徴を持ち、そのジャンルの真髄を捉えています。"
    
    # 楽器構成と相互作用について
    prominent_instruments = [k for k, v in analysis["instruments"].items() if v > 0.7]
    secondary_instruments = [k for k, v in analysis["instruments"].items() if 0.4 <= v < 0.7]
    
    instruments_description = ""
    
    if prominent_instruments:
        instruments_description += f"主要な楽器として{', '.join(prominent_instruments)}が特に目立ち"
        
        # 楽器ごとの役割や関係性
        if "ドラム" in prominent_instruments and "ベース" in prominent_instruments:
            instruments_description += "、リズムセクションが楽曲の骨格を強固に支え"
        elif "ドラム" in prominent_instruments:
            instruments_description += "、ドラムが楽曲の躍動感を生み出し"
        elif "ベース" in prominent_instruments:
            instruments_description += "、ベースが楽曲の土台を形成し"
        
        if "ギター" in prominent_instruments:
            if analysis["energy"] > 0.7:
                instruments_description += "、ギターが力強いリフとエネルギッシュなプレイで音楽を牽引し"
            else:
                instruments_description += "、ギターが繊細なアルペジオとメロディックなフレーズで彩りを加え"
        
        if "ピアノ" in prominent_instruments:
            instruments_description += "、ピアノが豊かなハーモニーと音楽性をもたらし"
        
        if "シンセサイザー" in prominent_instruments:
            instruments_description += "、シンセサイザーが現代的で革新的なサウンドスケープを創り出し"
        
        if "ボーカル" in prominent_instruments:
            if analysis["energy"] > 0.7:
                instruments_description += "、力強いボーカルが情感を伝え"
            else:
                instruments_description += "、表現力豊かなボーカルが物語を紡ぎ"
    
    if secondary_instruments:
        if instruments_description:
            instruments_description += f"ています。また、背景には{', '.join(secondary_instruments)}が補助的な役割を果たし、音のテクスチャーを豊かにしています。"
        else:
            instruments_description += f"{', '.join(secondary_instruments)}が調和的に組み合わさり、バランスの取れたサウンドを形成しています。"
    elif instruments_description:
        instruments_description += "ています。"
    
    if instruments_description:
        description += f" {instruments_description}"
    
    # セクション構造と音楽理論について
    structure_description = f"楽曲は約{analysis['sections']}つのセクションで構成されており、"
    
    # セクション数に基づく曲の複雑さ
    if analysis['sections'] <= 2:
        structure_description += "シンプルな構造ながらも効果的な楽曲構成を持ち、"
    elif analysis['sections'] <= 4:
        structure_description += "標準的なヴァース-コーラス形式に近い構造で、聴き手を自然に導き、"
    else:
        structure_description += "複数のセクションを持つ複雑な構造で、豊かな音楽的展開と変化に富み、"
    
    # アコースティック度と音響特性による追加説明
    sound_description = ""
    
    # アコースティック度合い
    if analysis["acousticness"] > 0.8:
        sound_description += "アコースティック楽器の自然な響きを最大限に活かし、オーガニックで温かみのある音響空間が特徴的です。"
    elif analysis["acousticness"] > 0.5:
        sound_description += "アコースティック要素を基調としながらも、適度な音響処理によって洗練された音像を持っています。"
    elif analysis["acousticness"] < 0.3:
        sound_description += "電子的な処理が施され、現代的かつ洗練されたスタジオプロダクションの特徴を備えています。"
    else:
        sound_description += "アコースティックと電子的な要素がバランス良く混ざり合い、多様な音色のレイヤーが重なり合った豊かな音響空間を形成しています。"
    
    description += f" {structure_description}{sound_description}"
    
    # ダンス性とリズム特性による追加説明
    if analysis["danceability"] > 0.8:
        description += " 極めて強いグルーヴ感を持ち、自然と体を動かしたくなるような魅力的なリズムパターンが特徴的です。"
    elif analysis["danceability"] > 0.6:
        description += " リズムが強調された心地よいグルーヴ感があり、ダンサブルな楽曲です。"
    
    # 感情と聴取シーンの提案
    mood_mapping = {
        "rock": "活力と解放感を与えてくれる",
        "pop": "親しみやすく心地よい雰囲気の",
        "jazz": "洗練された雰囲気と即興性を感じさせる",
        "classical": "情緒豊かで構造的な美しさを持つ",
        "electronic": "現代的で実験的な音響体験を提供する",
        "hip-hop": "都会的でリズミカルな表現力を持つ",
        "country": "素朴で物語性のある親しみやすい",
        "blues": "感情の機微を捉えた深みのある",
        "reggae": "リラックスした雰囲気と独特のリズム感を持つ",
        "metal": "力強さと技巧性が融合した"
    }
    
    if top_genre in mood_mapping:
        description += f" この音楽は{mood_mapping[top_genre]}曲で、{energy_feel}音楽性と{tempo_feel}リズムが融合した{energy_desc}サウンドスケープを創り出しています。"
    else:
        description += f" 全体として、この作品は{energy_feel}音楽性と{tempo_feel}リズムが融合した{energy_desc}サウンドスケープを創り出しています。"
    
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
        
        # 初期値を設定（すべてのフィールドにデフォルト値を設定）
        analysis_result = {
            # 基本音楽情報
            "tempo": 120.0,
            "beat_strength": 0.5,
            "tempo_stability": 0.5,
            "time_signature": "4/4",
            "key": "C major",
            "key_confidence": 0.5,
            
            # 音量とダイナミクス
            "mean_volume": 0.1,
            "max_volume": 0.2,
            "dynamic_range": 0.5,
            "volume_change_rate": 0.1,
            "energy": 0.5,
            
            # スペクトル特性
            "spectral_centroid": 1000.0,
            "spectral_bandwidth": 1000.0,
            "spectral_rolloff": 2000.0,
            "spectral_flatness": 0.01,
            "zero_crossing_rate": 0.1,
            "brightness": 0.5,
            "harmonicity": 0.5,
            "roughness": 0.5,
            
            # リズムと動き
            "danceability": 0.5,
            "attack_strength": 0.5,
            "acousticness": 0.5,
            
            # 構造
            "sections": 3,
            "chorus_likelihood": 0.5,
            
            # 楽器
            "instruments": {
                "ピアノ": 0.3,
                "ギター": 0.3,
                "ドラム": 0.3,
                "ベース": 0.3,
                "シンセサイザー": 0.3,
                "ボーカル": 0.3
            }
        }
        
        try:
            #===== 基本音楽特性 =====
            
            try:
                # テンポ（BPM）推定
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                # NumPy 2.0互換: 配列から要素を取り出してからfloatに変換
                if hasattr(tempo, '__len__') and len(tempo) > 0:
                    analysis_result["tempo"] = float(tempo[0])
                else:
                    analysis_result["tempo"] = float(tempo)
                
                # ビート強度を計算
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
                analysis_result["beat_strength"] = float(np.mean(pulse))
                
                # テンポ安定性（ビート間隔のばらつき）
                if len(beats) > 1:
                    beat_intervals = np.diff(beats)
                    analysis_result["tempo_stability"] = 1.0 - min(float(np.std(beat_intervals) / np.mean(beat_intervals)), 1.0)
                
                # 拍子推定（簡易版：強拍の間隔から4/4, 3/4などを推定）
                onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
                _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr)
                if len(beat_frames) > 4:
                    # 強拍検出（スペクトル強度に基づく簡易的な方法）
                    onset_strength = onset_envelope[beat_frames]
                    # 一定の間隔（4拍子か3拍子かなど）で強拍が来るかを確認
                    if len(onset_strength) > 8:
                        four_pattern = np.mean([onset_strength[i] for i in range(0, len(onset_strength) - 4, 4)])
                        three_pattern = np.mean([onset_strength[i] for i in range(0, len(onset_strength) - 3, 3)])
                        if four_pattern > three_pattern:
                            analysis_result["time_signature"] = "4/4"
                        else:
                            analysis_result["time_signature"] = "3/4"
                
                # キー推定
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                key_indices = np.sum(chroma, axis=1)
                key_index = np.argmax(key_indices)
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key = keys[key_index]
                
                # メジャー/マイナー判定
                major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
                minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
                
                # 循環シフトして相関を計算
                major_corrs = [np.corrcoef(np.roll(major_profile, i), key_indices)[0, 1] for i in range(12)]
                minor_corrs = [np.corrcoef(np.roll(minor_profile, i), key_indices)[0, 1] for i in range(12)]
                
                # 最も相関の高いモードを選択
                max_major_corr = max(major_corrs)
                max_minor_corr = max(minor_corrs)
                if max_major_corr > max_minor_corr:
                    mode = "major"
                    analysis_result["key_confidence"] = float(max_major_corr)
                else:
                    mode = "minor"
                    analysis_result["key_confidence"] = float(max_minor_corr)
                
                analysis_result["key"] = f"{key} {mode}"
            except Exception as e:
                debug_print(f"Error calculating basic music features: {str(e)}")
            
            #===== 音量とダイナミクス =====
            
            try:
                # 音量分析
                rms_values = librosa.feature.rms(y=y)[0]
                analysis_result["mean_volume"] = float(np.mean(rms_values))
                analysis_result["max_volume"] = float(np.max(rms_values))
                
                # エネルギー計算
                energy = float(analysis_result["mean_volume"]) / 0.1  # 正規化（典型的なRMS値に基づく）
                analysis_result["energy"] = min(max(energy, 0.0), 1.0)  # 0〜1の範囲に収める
                
                # ダイナミックレンジ（音量の変化幅）
                if analysis_result["max_volume"] > 0:
                    analysis_result["dynamic_range"] = float(np.std(rms_values) / analysis_result["max_volume"])
                    
                # 音量変化率（音量の時間的変化）
                volume_changes = np.abs(np.diff(rms_values))
                analysis_result["volume_change_rate"] = float(np.mean(volume_changes) / (analysis_result["mean_volume"] + 1e-6))
            except Exception as e:
                debug_print(f"Error calculating volume metrics: {str(e)}")
            
            #===== スペクトル特性 =====
            
            try:
                # スペクトル特性
                analysis_result["spectral_centroid"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
                analysis_result["spectral_bandwidth"] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
                analysis_result["spectral_rolloff"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
                analysis_result["spectral_flatness"] = float(librosa.feature.spectral_flatness(y=y).mean())
                analysis_result["zero_crossing_rate"] = float(librosa.feature.zero_crossing_rate(y).mean())
                
                # 音色の明るさ指標（高周波成分の比率）
                # 8kHz以上の周波数成分の割合を計算
                stft = np.abs(librosa.stft(y))
                freqs = librosa.fft_frequencies(sr=sr)
                high_freq_idx = np.where(freqs >= 8000)[0]
                if len(high_freq_idx) > 0 and stft.sum() > 0:
                    analysis_result["brightness"] = float(stft[high_freq_idx].sum() / stft.sum())
                
                # 調和性（倍音構造の強さ）
                harmonic, percussive = librosa.decompose.hpss(y)
                if np.sum(np.abs(y)) > 0:
                    analysis_result["harmonicity"] = float(np.sum(np.abs(harmonic)) / np.sum(np.abs(y)))
                    analysis_result["roughness"] = 1.0 - analysis_result["harmonicity"]
            except Exception as e:
                debug_print(f"Error calculating spectral features: {str(e)}")
            
            #===== リズムと時間特性 =====
            
            try:
                # ダンス性（リズムの規則性に基づく）
                analysis_result["danceability"] = min(max(float(np.mean(pulse)) * 0.5, 0.0), 1.0)
                
                # アタック特性（音の立ち上がりの強さ）
                # オンセット（音の始まり）の強度を計算
                onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
                onset_times = librosa.times_like(onset_strength, sr=sr)
                peaks = librosa.util.peak_pick(onset_strength, 3, 3, 3, 5, 0.5, 10)
                if len(peaks) > 0:
                    analysis_result["attack_strength"] = float(np.mean(onset_strength[peaks]))
                
                # アコースティック度（スペクトル特性に基づく）
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                acousticness = 1.0 - (float(np.mean(spectral_contrast[1:])) / 50.0)  # 高域のコントラストが低いほどアコースティック
                analysis_result["acousticness"] = min(max(acousticness, 0.0), 1.0)
            except Exception as e:
                debug_print(f"Error calculating rhythm features: {str(e)}")
            
            #===== 構造分析 =====
            try:
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
                    for i in range(1, len(boundaries)):
                        if boundaries[i] - filtered_boundaries[-1] > sr * 5:  # 5秒以上離れていれば新しいセクション
                            filtered_boundaries.append(boundaries[i])
                    analysis_result["sections"] = len(filtered_boundaries) + 1
                else:
                    analysis_result["sections"] = 1
                
                # セクション内の繰り返しパターン検出（コーラス検出の簡易版）
                # 類似した領域のカウント
                similarity_matrix = librosa.segment.recurrence_matrix(mfccs, mode='affinity')
                analysis_result["chorus_likelihood"] = float(np.mean(similarity_matrix))  # 繰り返しの多さを示す値
            except Exception as e:
                debug_print(f"Error in structure analysis: {str(e)}")
                # デフォルト値は既に設定済み
                
            #===== 楽器推定 =====
            
            try:
                # 新しい instruments 辞書を作成
                instruments = {
                    "ピアノ": 0.0,
                    "ギター": 0.0,
                    "ドラム": 0.0,
                    "ベース": 0.0,
                    "シンセサイザー": 0.0,
                    "ボーカル": 0.0
                }
                
                # ドラムの検出
                drums_strength = float(librosa.feature.rms(y=percussive).mean() / librosa.feature.rms(y=y).mean())
                instruments["ドラム"] = min(max(drums_strength * 1.5, 0.0), 1.0)
                
                # ボーカルの検出
                melspec = librosa.feature.melspectrogram(y=harmonic, sr=sr)
                vocal_range = float(np.mean(melspec[30:70, :]) / np.mean(melspec))  # ボーカル周波数帯の強さ
                instruments["ボーカル"] = min(max(vocal_range * 2.0 - 0.3, 0.0), 1.0)
                
                # ベースの検出
                bass_range = float(np.mean(melspec[:20, :]) / np.mean(melspec))
                instruments["ベース"] = min(max(bass_range * 2.0 - 0.2, 0.0), 1.0)
                
                # ギター/ピアノ/シンセの推定
                if analysis_result["spectral_flatness"] > 0.01:  # シンセの特徴
                    instruments["シンセサイザー"] = min(analysis_result["spectral_flatness"] * 50, 1.0)
                
                if 1500 < analysis_result["spectral_centroid"] < 3000 and analysis_result["spectral_bandwidth"] < 2000:
                    instruments["ピアノ"] = 0.7
                    
                if 800 < analysis_result["spectral_centroid"] < 2500:
                    instruments["ギター"] = 0.6
                    
                # 楽器推定結果を analysis_result に代入
                analysis_result["instruments"] = instruments
            except Exception as e:
                debug_print(f"Error calculating instrument detection: {str(e)}")
            
            # 基本音楽特性の計算
            try:
                # テンポ（BPM）推定
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                # NumPy 2.0互換: 配列から要素を取り出してからfloatに変換
                if hasattr(tempo, '__len__') and len(tempo) > 0:
                    analysis_result["tempo"] = float(tempo[0])
                else:
                    analysis_result["tempo"] = float(tempo)
                
                # ビート強度を計算
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
                analysis_result["beat_strength"] = float(np.mean(pulse))
                
                # テンポ安定性（ビート間隔のばらつき）
                if len(beats) > 1:
                    beat_intervals = np.diff(beats)
                    analysis_result["tempo_stability"] = 1.0 - min(float(np.std(beat_intervals) / np.mean(beat_intervals)), 1.0)
            except Exception as e:
                debug_print(f"Error calculating tempo metrics: {str(e)}")
                
            try:
                # 拍子推定
                onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
                _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr)
                if len(beat_frames) > 4:
                    # 強拍検出
                    onset_strength = onset_envelope[beat_frames]
                    if len(onset_strength) > 8:
                        four_pattern = np.mean([onset_strength[i] for i in range(0, len(onset_strength) - 4, 4)])
                        three_pattern = np.mean([onset_strength[i] for i in range(0, len(onset_strength) - 3, 3)])
                        analysis_result["time_signature"] = "4/4" if four_pattern > three_pattern else "3/4"
            except Exception as e:
                debug_print(f"Error calculating time signature: {str(e)}")
                        
            try:
                # キー推定
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                key_indices = np.sum(chroma, axis=1)
                key_index = np.argmax(key_indices)
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key = keys[key_index]
                
                # メジャー/マイナー判定
                major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
                minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
                
                # 循環シフトして相関を計算
                major_corrs = [np.corrcoef(np.roll(major_profile, i), key_indices)[0, 1] for i in range(12)]
                minor_corrs = [np.corrcoef(np.roll(minor_profile, i), key_indices)[0, 1] for i in range(12)]
                
                # 最も相関の高いモードを選択
                max_major_corr = max(major_corrs)
                max_minor_corr = max(minor_corrs)
                if max_major_corr > max_minor_corr:
                    mode = "major"
                    analysis_result["key_confidence"] = float(max_major_corr)
                else:
                    mode = "minor"
                    analysis_result["key_confidence"] = float(max_minor_corr)
                    
                analysis_result["key"] = f"{key} {mode}"
            except Exception as e:
                debug_print(f"Error calculating key: {str(e)}")
                
            try:
                # 音量分析
                rms_values = librosa.feature.rms(y=y)[0]
                analysis_result["mean_volume"] = float(np.mean(rms_values))
                analysis_result["max_volume"] = float(np.max(rms_values))
                
                # エネルギー計算
                energy = float(np.mean(rms_values)) / 0.1  # 正規化
                analysis_result["energy"] = min(max(energy, 0.0), 1.0)  # 0〜1の範囲に収める
                
                # ダイナミックレンジと音量変化率
                if analysis_result["max_volume"] > 0:
                    analysis_result["dynamic_range"] = float(np.std(rms_values) / analysis_result["max_volume"])
                    
                volume_changes = np.abs(np.diff(rms_values))
                analysis_result["volume_change_rate"] = float(np.mean(volume_changes) / (analysis_result["mean_volume"] + 1e-6))
            except Exception as e:
                debug_print(f"Error calculating volume metrics: {str(e)}")
                
            try:
                # スペクトル特性
                analysis_result["spectral_centroid"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
                analysis_result["spectral_bandwidth"] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
                analysis_result["spectral_rolloff"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
                analysis_result["spectral_flatness"] = float(librosa.feature.spectral_flatness(y=y).mean())
                analysis_result["zero_crossing_rate"] = float(librosa.feature.zero_crossing_rate(y).mean())
            except Exception as e:
                debug_print(f"Error calculating spectral features: {str(e)}")
                
            try:
                # 音色の明るさ指標
                stft = np.abs(librosa.stft(y))
                freqs = librosa.fft_frequencies(sr=sr)
                high_freq_idx = np.where(freqs >= 8000)[0]
                if len(high_freq_idx) > 0 and stft.sum() > 0:
                    analysis_result["brightness"] = float(stft[high_freq_idx].sum() / stft.sum())
            except Exception as e:
                debug_print(f"Error calculating brightness: {str(e)}")
                
            try:
                # 調和性（倍音構造の強さ）
                harmonic, percussive = librosa.decompose.hpss(y)
                if np.sum(np.abs(y)) > 0:
                    analysis_result["harmonicity"] = float(np.sum(np.abs(harmonic)) / np.sum(np.abs(y)))
                    analysis_result["roughness"] = 1.0 - analysis_result["harmonicity"]
                    
                # ダンス性とアタック
                analysis_result["danceability"] = min(max(float(np.mean(pulse)) * 0.5, 0.0), 1.0)
                
                # オンセット強度
                onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
                peaks = librosa.util.peak_pick(onset_strength, 3, 3, 3, 5, 0.5, 10)
                if len(peaks) > 0:
                    analysis_result["attack_strength"] = float(np.mean(onset_strength[peaks]))
                    
                # アコースティック度
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                analysis_result["acousticness"] = 1.0 - (float(np.mean(spectral_contrast[1:])) / 50.0)
                analysis_result["acousticness"] = min(max(analysis_result["acousticness"], 0.0), 1.0)
                
                # 楽器推定
                instruments = {}
                instruments["ドラム"] = min(max(float(librosa.feature.rms(y=percussive).mean() / librosa.feature.rms(y=y).mean()) * 1.5, 0.0), 1.0)
                
                # 他の楽器の検出
                melspec = librosa.feature.melspectrogram(y=harmonic, sr=sr)
                
                # ボーカル検出
                vocal_range = float(np.mean(melspec[30:70, :]) / np.mean(melspec))
                instruments["ボーカル"] = min(max(vocal_range * 2.0 - 0.3, 0.0), 1.0)
                
                # ベース検出
                bass_range = float(np.mean(melspec[:20, :]) / np.mean(melspec))
                instruments["ベース"] = min(max(bass_range * 2.0 - 0.2, 0.0), 1.0)
                
                # シンセ、ピアノ、ギター検出
                if analysis_result["spectral_flatness"] > 0.01:
                    instruments["シンセサイザー"] = min(analysis_result["spectral_flatness"] * 50, 1.0)
                else:
                    instruments["シンセサイザー"] = 0.3
                    
                if 1500 < analysis_result["spectral_centroid"] < 3000 and analysis_result["spectral_bandwidth"] < 2000:
                    instruments["ピアノ"] = 0.7
                else:
                    instruments["ピアノ"] = 0.3
                    
                if 800 < analysis_result["spectral_centroid"] < 2500:
                    instruments["ギター"] = 0.6
                else:
                    instruments["ギター"] = 0.3
                    
                analysis_result["instruments"] = instruments
            except Exception as e:
                debug_print(f"Error calculating advanced features: {str(e)}")
                
            return analysis_result
            
        except Exception as e:
            debug_print(f"Error analyzing audio: {str(e)}")
            # エラーが発生した場合でも、可能な限りのデータを返す
            return analysis_result
    
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
            description = f"この楽曲は{tempo_desc}テンポ（{tempo:.1f}BPM）の{energy_desc}{top_genre}曲です。"
            
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
            
            # ジャンルの混合について（より詳細に）
            if second_genre and genres[1]["confidence"] > 0.3:
                if third_genre and genres[2]["confidence"] > 0.2:
                    description += f" 主に{top_genre}をベースとしながらも、{second_genre}と{third_genre}の要素が融合した多面的な音楽性を持っています。"
                else:
                    description += f" {second_genre}の影響を強く受けた{top_genre}スタイルが特徴的です。"
            else:
                description += f" 典型的な{top_genre}の特徴を持ち、そのジャンルの真髄を捉えています。"
            
            # 楽器構成と相互作用について
            prominent_instruments = [k for k, v in analysis["instruments"].items() if v > 0.7]
            secondary_instruments = [k for k, v in analysis["instruments"].items() if 0.4 <= v < 0.7]
            
            instruments_description = ""
            
            if prominent_instruments:
                instruments_description += f"主要な楽器として{', '.join(prominent_instruments)}が特に目立ち"
                
                # 楽器ごとの役割や関係性
                if "ドラム" in prominent_instruments and "ベース" in prominent_instruments:
                    instruments_description += "、リズムセクションが楽曲の骨格を強固に支え"
                elif "ドラム" in prominent_instruments:
                    instruments_description += "、ドラムが楽曲の躍動感を生み出し"
                elif "ベース" in prominent_instruments:
                    instruments_description += "、ベースが楽曲の土台を形成し"
                
                if "ギター" in prominent_instruments:
                    if analysis["energy"] > 0.7:
                        instruments_description += "、ギターが力強いリフとエネルギッシュなプレイで音楽を牽引し"
                    else:
                        instruments_description += "、ギターが繊細なアルペジオとメロディックなフレーズで彩りを加え"
                
                if "ピアノ" in prominent_instruments:
                    instruments_description += "、ピアノが豊かなハーモニーと音楽性をもたらし"
                
                if "シンセサイザー" in prominent_instruments:
                    instruments_description += "、シンセサイザーが現代的で革新的なサウンドスケープを創り出し"
                
                if "ボーカル" in prominent_instruments:
                    if analysis["energy"] > 0.7:
                        instruments_description += "、力強いボーカルが情感を伝え"
                    else:
                        instruments_description += "、表現力豊かなボーカルが物語を紡ぎ"
            
            if secondary_instruments:
                if instruments_description:
                    instruments_description += f"ています。また、背景には{', '.join(secondary_instruments)}が補助的な役割を果たし、音のテクスチャーを豊かにしています。"
                else:
                    instruments_description += f"{', '.join(secondary_instruments)}が調和的に組み合わさり、バランスの取れたサウンドを形成しています。"
            elif instruments_description:
                instruments_description += "ています。"
            
            if instruments_description:
                description += f" {instruments_description}"
            
            # セクション構造と音楽理論について
            structure_description = f"楽曲は約{analysis['sections']}つのセクションで構成されており、"
            
            # セクション数に基づく曲の複雑さ
            if analysis['sections'] <= 2:
                structure_description += "シンプルな構造ながらも効果的な楽曲構成を持ち、"
            elif analysis['sections'] <= 4:
                structure_description += "標準的なヴァース-コーラス形式に近い構造で、聴き手を自然に導き、"
            else:
                structure_description += "複数のセクションを持つ複雑な構造で、豊かな音楽的展開と変化に富み、"
            
            # テンポ安定性に関するコメント（あれば）
            if "tempo_stability" in analysis:
                if analysis["tempo_stability"] > 0.8:
                    structure_description += "安定したテンポキープが特徴で、"
                elif analysis["tempo_stability"] < 0.5:
                    structure_description += "テンポの緩急が表現力を高め、"
            
            # コード進行の複雑さのヒント（キーコンフィデンスが指標になる場合）
            if "key_confidence" in analysis:
                if analysis["key_confidence"] > 0.8:
                    structure_description += "調性が明確に確立されたシンプルなコード進行が使われています。"
                elif analysis["key_confidence"] < 0.5:
                    structure_description += "複雑で予想外のコード進行が音楽的な深みを生み出しています。"
                else:
                    structure_description += "バランスの取れたハーモニー展開が楽曲を支えています。"
            else:
                structure_description += "全体的にまとまりのある構成となっています。"
            
            description += f" {structure_description}"
            
            # アコースティック度と音響特性による追加説明
            sound_description = ""
            
            # アコースティック度合い
            if analysis["acousticness"] > 0.8:
                sound_description += "アコースティック楽器の自然な響きを最大限に活かし、オーガニックで温かみのある音響空間が特徴的です。"
            elif analysis["acousticness"] > 0.5:
                sound_description += "アコースティック要素を基調としながらも、適度な音響処理によって洗練された音像を持っています。"
            elif analysis["acousticness"] < 0.3:
                if "spectral_flatness" in analysis and analysis["spectral_flatness"] > 0.01:
                    sound_description += "電子的な処理とシンセサイザーが多用された現代的なサウンドデザインが特徴的で、革新的な音響体験を提供しています。"
                else:
                    sound_description += "電子的な処理が施され、現代的かつ洗練されたスタジオプロダクションの特徴を備えています。"
            else:
                sound_description += "アコースティックと電子的な要素がバランス良く混ざり合い、多様な音色のレイヤーが重なり合った豊かな音響空間を形成しています。"
            
            # 周波数特性に関する追加情報
            if "spectral_centroid" in analysis:
                if analysis["spectral_centroid"] > 5000:
                    sound_description += " 高域が豊かに表現され、明るく透明感のある音色が特徴です。"
                elif analysis["spectral_centroid"] < 1000:
                    sound_description += " 低域が充実した重厚感のある音色で、深みのある響きを持っています。"
            
            description += f" {sound_description}"
            
            # ダンス性とリズム特性による追加説明
            if analysis["danceability"] > 0.8:
                description += " 極めて強いグルーヴ感を持ち、自然と体を動かしたくなるような魅力的なリズムパターンが特徴的です。"
            elif analysis["danceability"] > 0.6:
                description += " リズムが強調された心地よいグルーヴ感があり、ダンサブルな楽曲です。"
            
            # リズム複雑性（ある場合）
            if "attack_strength" in analysis and analysis["attack_strength"] > 0.5:
                description += " リズムにアクセントと変化が豊富で、単調さを感じさせない工夫が施されています。"
            
            # 感情と聴取シーンの提案
            mood_mapping = {
                ("rock", True): "活力とエネルギーを与えてくれる曲で、運動や活動的なシーンに最適です",
                ("rock", False): "内省的な雰囲気を持つロック曲で、くつろいだ時間や夜のドライブに合います",
                ("pop", True): "ポジティブで前向きな気持ちにさせてくれる曲で、パーティーやソーシャルな場面に適しています",
                ("pop", False): "メロディアスで心地よい雰囲気のポップ曲で、リラックスしたい時間に最適です",
                ("jazz", True): "洗練された活気あるジャズで、おしゃれなカフェやディナーの背景音楽として理想的です",
                ("jazz", False): "落ち着きと深みのあるジャズナンバーで、読書や思考を深める時間に合います",
                ("classical", True): "力強さと躍動感のあるクラシック音楽で、集中力を高めたい時に効果的です",
                ("classical", False): "優美で繊細なクラシック作品で、穏やかな瞑想や睡眠前の時間に最適です",
                ("electronic", True): "刺激的で革新的な電子音楽で、現代的な空間やナイトライフに溶け込みます",
                ("electronic", False): "アンビエント要素のある電子音楽で、作業中のバックグラウンドミュージックとして機能します",
                ("hip-hop", True): "力強いビートとエネルギッシュなフローが特徴のヒップホップで、自信と活力を与えてくれます",
                ("hip-hop", False): "思慮深いリリックと落ち着いたビートのヒップホップで、内省的な時間に適しています",
                ("country", True): "明るく活気のあるカントリーミュージックで、ロードトリップや屋外活動に最適です",
                ("country", False): "物語性豊かな静かなカントリーソングで、穏やかな夕暮れや静かな時間に合います",
                ("blues", True): "リズミカルで表現力豊かなブルースで、感情を解放したい時に響きます",
                ("blues", False): "深い感情と静謐さを兼ね備えたブルースで、個人的な内省の時間に寄り添います",
                ("reggae", True): "陽気でリズミカルなレゲエで、ビーチやリラックスした社交の場に最適です",
                ("reggae", False): "ダブ要素の強いレイドバックしたレゲエで、心を落ち着かせたい時間に合います",
                ("metal", True): "テクニカルで力強いメタルで、激しい運動や感情の発散に効果的です",
                ("metal", False): "複雑な構造と深い雰囲気を持つメタル曲で、没入型の音楽体験を提供します"
            }
            
            # 高エネルギーかどうか
            is_energetic = energy > 0.5
            
            if (top_genre, is_energetic) in mood_mapping:
                description += f" この音楽は{mood_mapping[(top_genre, is_energetic)]}。"
            
            # 全体のまとめとなる一文を追加
            description += f" 全体として、この{top_genre}作品は{energy_feel}音楽性と{tempo_feel}リズムが融合した{energy_desc}サウンドスケープを創り出しています。"
                
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
