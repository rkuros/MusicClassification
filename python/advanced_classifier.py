#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import numpy as np
import argparse
from datetime import datetime

# 共通ユーティリティをインポート
from utils import numpy_to_python_type, debug_print, generate_mock_waveform, generate_mock_analysis

# スクリプト開始時のデバッグ情報
debug_print(f"Advanced Genre Classifier started at {datetime.now()}")

# 拡張ジャンル一覧
EXTENDED_GENRES = [
    'rock', 'pop', 'jazz', 'classical', 'electronic',
    'hip-hop', 'country', 'blues', 'reggae', 'metal',
    'dance', 'funk', 'soul', 'folk', 'ambient', 'indie'
]

# 音楽属性一覧
MUSIC_ATTRIBUTES = [
    'high_energy', 'chill', 'acoustic', 'instrumental', 'vocal', 
    'fast_tempo', 'slow_tempo', 'upbeat', 'melancholic'
]

# 必須ライブラリのインポートを試みる
USE_MOCK = True
HAS_PYTORCH = False

try:
    debug_print("Importing libraries for advanced analysis...")
    print("Trying to import librosa...", file=sys.stderr)
    
    import librosa
    debug_print(f"Librosa imported successfully, version: {librosa.__version__}")
    print(f"Librosa imported successfully, version: {librosa.__version__}", file=sys.stderr)
    
    print("Trying to import sklearn modules...", file=sys.stderr)
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    print("Trying to import joblib...", file=sys.stderr)
    import joblib
    print("sklearn and joblib imported successfully", file=sys.stderr)
    
    # PyTorchもインポートを試みる
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchaudio
        HAS_PYTORCH = True
        debug_print(f"PyTorch imported successfully, version: {torch.__version__}")
    except ImportError:
        HAS_PYTORCH = False
        debug_print("PyTorch not available, will use RandomForest only")
    
    # ライブラリが正常にインポートできたかチェック
    USE_MOCK = False
    debug_print("All required libraries for advanced analysis successfully imported.")
    print("必要なライブラリがインストールされています。高度分析モードで実行します", file=sys.stderr)
except ImportError as e:
    debug_print(f"Import error: {e}")
    print("必要なライブラリがインストールされていません。モックモードで実行します", file=sys.stderr)
except Exception as e:
    debug_print(f"Unexpected error during imports: {e}")
    print(f"予期しないエラーが発生しました: {e}", file=sys.stderr)

# モデルファイルのパス
ADVANCED_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'advanced_model.pkl')
ADVANCED_SCALER_PATH = os.path.join(os.path.dirname(__file__), 'advanced_scaler.pkl')
ADVANCED_NN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'advanced_nn_model.pt')

# PyTorchによるニューラルネットワークモデルの定義
if HAS_PYTORCH:
    class MusicGenreClassifier(nn.Module):
        """
        音楽ジャンル分類のための多層ニューラルネットワーク
        """
        def __init__(self, input_size, num_classes):
            """
            MusicGenreClassifierの初期化
            
            Parameters:
            -----------
            input_size : int
                入力特徴量の次元数
            num_classes : int
                分類するジャンルの数
            """
            super(MusicGenreClassifier, self).__init__()
            
            # ネットワークレイヤーの定義
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),  # 過学習防止
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),  # 過学習防止
                nn.Linear(64, num_classes)
            )
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x):
            """
            順伝播
            
            Parameters:
            -----------
            x : torch.Tensor
                入力特徴量
                
            Returns:
            --------
            torch.Tensor
                予測クラス確率
            """
            logits = self.model(x)
            return self.softmax(logits)

def generate_mock_description(genres, analysis, attributes=None):
    """
    モックの楽曲説明文を生成する（属性情報含む拡張版）
    
    Parameters:
    -----------
    genres : list
        ジャンルと確信度のリスト
    analysis : dict
        楽曲分析情報
    attributes : dict, optional
        検出された音楽属性情報
        
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
    
    # 属性情報がある場合は追加
    if attributes:
        attribute_desc = []
        
        if attributes.get("high_energy"):
            attribute_desc.append("高いエネルギー感")
        if attributes.get("chill"):
            attribute_desc.append("リラックス感")
        if attributes.get("acoustic"):
            attribute_desc.append("アコースティックな質感")
        if attributes.get("instrumental"):
            attribute_desc.append("インストゥルメンタル的な特性")
        if attributes.get("vocal"):
            attribute_desc.append("ヴォーカル中心の構成")
        if attributes.get("fast_tempo"):
            attribute_desc.append("テンポの速さ")
        if attributes.get("slow_tempo"):
            attribute_desc.append("ゆったりとしたテンポ")
        if attributes.get("upbeat"):
            attribute_desc.append("明るいアップビート感")
        if attributes.get("melancholic"):
            attribute_desc.append("物悲しいメランコリックな雰囲気")
        
        if attribute_desc:
            description += f" 特徴的な属性として{', '.join(attribute_desc)}が検出されました。"
    
    # 調性の感情的な特徴
    if key_mode.lower() == "major":
        description += f"主要キーは{key_note} メジャーで、明るく開放的な響きが特徴的です。"
    elif key_mode.lower() == "minor":
        description += f"主要キーは{key_note} マイナーで、深みと感情的な響きが特徴的です。"
    else:
        description += f"主要キーは{key}です。"
    
    # 音楽的文脈・歴史的背景（拡張ジャンルにも対応）
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
        "metal": "激しいエネルギーと技巧的な演奏が特徴の",
        "dance": "踊りやすいリズムと躍動感あふれる",
        "funk": "グルーヴ感とシンコペーションが特徴的な",
        "soul": "感情表現の豊かさと深みのある",
        "folk": "物語性と自然な音響感を大切にした",
        "ambient": "空間的な広がりと穏やかな雰囲気の",
        "indie": "独自性と実験性を兼ね備えた"
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
        "metal": "力強さと技巧性が融合した", 
        "dance": "躍動感あふれるダンスフロアを彩る",
        "funk": "独特のグルーヴ感と躍動感に満ちた",
        "soul": "感情の機微を表現した深みのある",
        "folk": "ストーリーテリングと素朴な表現が魅力の",
        "ambient": "心地よい没入感と空間的広がりを持つ",
        "indie": "独創的なアプローチと個性が光る"
    }
    
    if top_genre in mood_mapping:
        description += f" この音楽は{mood_mapping[top_genre]}曲で、{energy_feel}音楽性と{tempo_feel}リズムが融合した{energy_desc}サウンドスケープを創り出しています。"
    else:
        description += f" 全体として、この作品は{energy_feel}音楽性と{tempo_feel}リズムが融合した{energy_desc}サウンドスケープを創り出しています。"
    
    return description

def generate_mock_results():
    """
    モックのジャンル分類結果を生成する（高度分析用）
    
    Returns:
    --------
    results : dict
        ジャンル分類、波形、分析結果を含む辞書
    """
    debug_print("Using mock mode - generating advanced analysis results")
    
    # ジャンルリスト（拡張版）
    genres_list = EXTENDED_GENRES
    
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
    
    # 音楽属性を生成
    attributes = {
        "high_energy": analysis["energy"] > 0.8,
        "chill": analysis["energy"] < 0.4 and analysis.get("tempo", 120) < 100,
        "acoustic": analysis["acousticness"] > 0.7,
        "instrumental": analysis["instruments"]["ボーカル"] < 0.3,
        "vocal": analysis["instruments"]["ボーカル"] > 0.7,
        "fast_tempo": analysis["tempo"] > 140,
        "slow_tempo": analysis["tempo"] < 80,
        "upbeat": analysis["energy"] > 0.6 and analysis["danceability"] > 0.6,
        "melancholic": "minor" in analysis["key"] and analysis["energy"] < 0.5
    }
    
    # 総合的な説明を生成
    description = generate_mock_description(genres, analysis, attributes)
    
    return {
        "genres": genres,
        "waveform": waveform,
        "analysis": analysis,
        "attributes": attributes,
        "description": description
    }

# 実際の分析機能 (librosaが必要)
if not USE_MOCK:
    def detect_attributes(y, sr, analysis):
        """
        音楽属性を検出する関数
        
        Parameters:
        -----------
        y : numpy.ndarray
            オーディオ信号
        sr : int
            サンプリングレート
        analysis : dict
            楽曲分析情報
            
        Returns:
        --------
        attributes : dict
            検出された音楽属性
        """
        try:
            # キーモードの取得（majorかminorか）
            key_mode = "unknown"
            if "key" in analysis and " " in analysis["key"]:
                key_parts = analysis["key"].split(" ")
                if len(key_parts) > 1:
                    key_mode = key_parts[1].lower()
            
            # エネルギー関連の属性
            high_energy = analysis["energy"] > 0.8
            chill = analysis["energy"] < 0.4 and analysis.get("tempo", 120) < 100
            
            # 音響特性関連の属性
            acoustic = analysis.get("acousticness", 0) > 0.7
            
            # ボーカル関連の属性
            instrumental = analysis.get("instruments", {}).get("ボーカル", 0) < 0.3
            vocal = analysis.get("instruments", {}).get("ボーカル", 0) > 0.7
            
            # テンポ関連の属性
            fast_tempo = analysis.get("tempo", 0) > 140
            slow_tempo = analysis.get("tempo", 0) < 80
            
            # 感情関連の属性
            upbeat = analysis["energy"] > 0.6 and analysis.get("danceability", 0) > 0.6
            melancholic = key_mode == "minor" and analysis["energy"] < 0.5
            
            attributes = {
                "high_energy": high_energy,
                "chill": chill,
                "acoustic": acoustic,
                "instrumental": instrumental,
                "vocal": vocal,
                "fast_tempo": fast_tempo,
                "slow_tempo": slow_tempo,
                "upbeat": upbeat,
                "melancholic": melancholic
            }
            
            return attributes
        except Exception as e:
            debug_print(f"Error detecting attributes: {str(e)}")
            # エラー時はデフォルト値を返す
            return {attr: False for attr in MUSIC_ATTRIBUTES}
    
    def calculate_danceability(y, sr):
        """
        ダンス性を計算する
        
        Parameters:
        -----------
        y : numpy.ndarray
            オーディオ信号
        sr : int
            サンプリングレート
            
        Returns:
        --------
        danceability : float
            ダンス性の値 (0～1)
        """
        try:
            # オンセット検出 (ビートの立ち上がり)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # パルス強度の測定 (ビートの強さ)
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
            pulse_mean = pulse.mean()
            
            # テンポの検出
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # ダンス性を計算 - テンポとパルス強度から
            # 理想的なダンスのテンポ範囲を定義 (90-140 BPM)
            tempo_factor = 0.0
            if 90 <= tempo <= 140:  # ダンスに適したテンポ範囲
                tempo_factor = 1.0 - abs((tempo - 115) / 50)  # 115 BPMを中心に正規化
            else:
                # 範囲外の場合は距離に応じて減衰
                tempo_factor = max(0, 1.0 - abs((tempo - 115) / 100))
            
            # パルス強度は0-1に正規化
            normalized_pulse = min(1.0, pulse_mean / 10.0)
            
            # ダンス性は、テンポの適切さとパルス強度を組み合わせて計算
            danceability = 0.6 * tempo_factor + 0.4 * normalized_pulse
            
            return min(1.0, max(0.0, danceability))  # 0～1に収める
        except Exception as e:
            debug_print(f"Error calculating danceability: {str(e)}")
            return 0.5  # エラー時はデフォルト値を返す
    
    def extract_features(y, sr):
        """
        音楽ファイルから特徴量を抽出する
        
        Parameters:
        -----------
        y : numpy.ndarray
            オーディオ信号
        sr : int
            サンプリングレート
            
        Returns:
        --------
        features : numpy.ndarray
            抽出された特徴量ベクトル
        """
        try:
            debug_print("Extracting advanced features...")
            features_list = []
            
            # MFCC - モデルに合わせて15次元に調整
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
            mfcc_mean = np.array([float(val) for val in mfcc.mean(axis=1)])
            mfcc_var = np.array([float(val) for val in mfcc.var(axis=1)])
            features_list.extend(mfcc_mean)
            features_list.extend(mfcc_var)
            
            # クロマグラム特徴量
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.array([float(val) for val in chroma.mean(axis=1)])
            chroma_var = np.array([float(val) for val in chroma.var(axis=1)])
            features_list.extend(chroma_mean)
            features_list.extend(chroma_var)
            
            # スペクトル特徴量
            centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
            rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
            flatness = float(librosa.feature.spectral_flatness(y=y).mean())
            
            features_list.append(centroid)
            features_list.append(bandwidth)
            features_list.append(rolloff)
            features_list.append(flatness)
            
            # テンポ・リズム特徴量
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # Fix numpy deprecation warning by using item() to extract scalar value
            if isinstance(tempo, np.ndarray):
                features_list.append(float(tempo.item()))
            else:
                features_list.append(float(tempo))
            
            # ゼロ交差率
            zero_crossing = float(librosa.feature.zero_crossing_rate(y).mean())
            features_list.append(zero_crossing)
            
            # 特徴量をnumpy配列に変換 - 明示的に型変換して均一な配列を保証
            features = np.array(features_list, dtype=np.float64)
            
            # 特徴量が60個になるように調整
            if len(features) > 60:
                debug_print(f"Truncating features from {len(features)} to 60")
                features = features[:60]  # 60個に切り詰める
            elif len(features) < 60:
                debug_print(f"Padding features from {len(features)} to 60")
                # 足りない場合はゼロで埋める
                padding = np.zeros(60 - len(features))
                features = np.concatenate([features, padding])
            
            debug_print(f"Final feature count: {len(features)}")
            return features
        except Exception as e:
            debug_print(f"Error extracting features: {str(e)}")
            # エラー時はランダム特徴量を返す
            return np.random.random(60)
    
    def predict_genre(features):
        """
        特徴量からジャンルを予測する
        
        Parameters:
        -----------
        features : numpy.ndarray
            特徴量ベクトル
            
        Returns:
        --------
        genres : list
            ジャンル予測結果のリスト
        """
        genres = []
        
        try:
            debug_print("Predicting genre...")
            
            # RandomForestによる予測
            if (os.path.exists(ADVANCED_MODEL_PATH) and os.path.getsize(ADVANCED_MODEL_PATH) > 0 and 
                os.path.exists(ADVANCED_SCALER_PATH) and os.path.getsize(ADVANCED_SCALER_PATH) > 0):
                try:
                    print(f"Loading model from {ADVANCED_MODEL_PATH} ({os.path.getsize(ADVANCED_MODEL_PATH)} bytes)", file=sys.stderr)
                    print(f"Loading scaler from {ADVANCED_SCALER_PATH} ({os.path.getsize(ADVANCED_SCALER_PATH)} bytes)", file=sys.stderr)
                    
                    # スケーラーとモデルを読み込む
                    scaler = joblib.load(ADVANCED_SCALER_PATH)
                    model = joblib.load(ADVANCED_MODEL_PATH)
                    
                    # 特徴量の次元数を確認
                    expected_features = scaler.n_features_in_
                    if features.shape[0] != expected_features:
                        debug_print(f"Feature count mismatch: Model expects {expected_features} features, but got {features.shape[0]} features")
                        raise ValueError(f"X has {features.shape[0]} features, but StandardScaler is expecting {expected_features} features as input.")
                        
                    # 特徴量のスケーリング
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # モデル予測
                    debug_print("Predicting with RandomForest model...")
                    probabilities = model.predict_proba(features_scaled)[0]
                    
                    # ジャンルと確信度のマッピング
                    for i, prob in enumerate(probabilities):
                        # インデックスがジャンル数を超えないようにチェック
                        if i < len(EXTENDED_GENRES):
                            genres.append({
                                "name": EXTENDED_GENRES[i],
                                "confidence": float(prob)
                            })
                    
                except Exception as e:
                    debug_print(f"RandomForest prediction error: {e}")
            
            # PyTorchモデルの予測（利用可能な場合）
            if HAS_PYTORCH and os.path.exists(ADVANCED_NN_MODEL_PATH):
                try:
                    debug_print("PyTorch model found, attempting neural network prediction...")
                    # 学習済みモデルの読み込み
                    model = MusicGenreClassifier(input_size=features.shape[0], num_classes=len(EXTENDED_GENRES))
                    model.load_state_dict(torch.load(ADVANCED_NN_MODEL_PATH))
                    model.eval()
                    
                    # 特徴量をPyTorchテンソルに変換
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    
                    # 予測
                    with torch.no_grad():
                        probabilities = model(input_tensor).squeeze(0).numpy()
                    
                    # ニューラルネットワークの予測結果を加味（既存の結果と平均）
                    if genres:
                        debug_print("Combining RandomForest and NN predictions...")
                        # 既存のRandomForest予測がある場合は平均を取る
                        combined_genres = []
                        for i, genre_name in enumerate(EXTENDED_GENRES):
                            rf_prob = next((g["confidence"] for g in genres if g["name"] == genre_name), 0.0)
                            nn_prob = probabilities[i]
                            
                            # 平均確率を計算
                            avg_prob = (rf_prob + nn_prob) / 2.0
                            
                            combined_genres.append({
                                "name": genre_name,
                                "confidence": float(avg_prob)
                            })
                        genres = combined_genres
                    else:
                        debug_print("Using only NN predictions...")
                        # RandomForest予測がない場合はNNの結果のみを使用
                        for i, prob in enumerate(probabilities):
                            if i < len(EXTENDED_GENRES):
                                genres.append({
                                    "name": EXTENDED_GENRES[i],
                                    "confidence": float(prob)
                                })
                                
                except Exception as e:
                    debug_print(f"Neural network prediction error: {e}")
            
            # 結果がない場合（どちらのモデルも使えなかった場合）はランダム予測を行う
            if not genres:
                debug_print("No model available, using random predictions")
                confidences = [random.random() for _ in range(len(EXTENDED_GENRES))]
                total = sum(confidences)
                confidences = [c / total for c in confidences]
                
                for i, genre in enumerate(EXTENDED_GENRES):
                    genres.append({
                        "name": genre,
                        "confidence": confidences[i]
                    })
            
            # 確信度でソート（降順）
            genres = sorted(genres, key=lambda x: x["confidence"], reverse=True)
            genres = genres[:5]  # 上位5つの結果のみ
            
            return genres
        except Exception as e:
            debug_print(f"Error in predict_genre: {e}")
            # エラー時はランダム予測を返す
            return generate_mock_results()["genres"]
            
    def analyze_audio(file_path, y=None, sr=None):
        """
        音楽ファイルの詳細な分析を行う
        
        Parameters:
        -----------
        file_path : str
            音楽ファイルのパス
        y : numpy.ndarray, optional
            既に読み込まれているオーディオ信号
        sr : int, optional
            既に読み込まれているサンプリングレート
            
        Returns:
        --------
        analysis : dict
            楽曲分析情報
        """
        try:
            debug_print(f"Starting advanced audio analysis for: {file_path}")
            
            # 音声信号がまだ読み込まれていない場合は読み込む
            if y is None or sr is None:
                y, sr = librosa.load(file_path, mono=True, duration=30)
            
            # テンポ検出（より堅牢なアルゴリズムを使用）
            debug_print("Detecting tempo with advanced algorithm...")
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
            
            # キー検出（拡張版）
            debug_print("Detecting key with harmonic-percussive source separation...")
            # ハーモニック成分を抽出（より正確なキー検出のため）
            y_harmonic = librosa.effects.harmonic(y)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
            
            # よりロバストなキー検出のために複数フレームの平均を取る
            chroma_avg = np.mean(chroma, axis=1)
            key_idx = np.argmax(chroma_avg)
            keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            
            # メジャー/マイナー判定（拡張版）
            # 相対マイナーの3度と6度の音程関係を考慮
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # メジャーキーのクロマプロファイル
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # マイナーキーのクロマプロファイル
            
            major_correlation = np.zeros(12)
            minor_correlation = np.zeros(12)
            
            # 各キーについて相関を計算
            for i in range(12):
                # 循環シフトでプロファイルを各キーに合わせる
                rolled_major = np.roll(major_profile, i)
                rolled_minor = np.roll(minor_profile, i)
                
                # 相関係数を計算
                major_correlation[i] = np.corrcoef(chroma_avg, rolled_major)[0, 1]
                minor_correlation[i] = np.corrcoef(chroma_avg, rolled_minor)[0, 1]
            
            # 最も相関の高いキーとスケールを選択
            max_major_idx = np.argmax(major_correlation)
            max_minor_idx = np.argmax(minor_correlation)
            
            # メジャーとマイナーのどちらの相関が高いか比較
            if major_correlation[max_major_idx] > minor_correlation[max_minor_idx]:
                key = keys[max_major_idx]
                mode = "major"
            else:
                key = keys[max_minor_idx]
                mode = "minor"
            
            key_result = f"{key} {mode}"
            
            # エネルギー計算（拡張版）
            debug_print("Calculating energy with spectral features...")
            # スペクトログラムの計算
            S = np.abs(librosa.stft(y))
            
            # RMS エネルギー（全体）
            rms_energy = np.mean(librosa.feature.rms(S=S)[0])
            # スペクトル重心（音の明るさ）
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0])
            
            # 加重エネルギー値の計算
            energy = 0.7 * min(1.0, rms_energy * 10) + 0.3 * min(1.0, spectral_centroid / (sr/4))
            
            # 楽器の検出（拡張版）
            debug_print("Detecting instruments with spectral features...")
            instruments = {
                "ピアノ": 0.0,
                "ギター": 0.0,
                "ドラム": 0.0,
                "ベース": 0.0,
                "シンセサイザー": 0.0,
                "ボーカル": 0.0
            }
            
            # メル周波数変換
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # 周波数帯域ごとのエネルギー分布
            bands = {
                "sub_bass": np.mean(log_mel_spec[:5, :]),         # ~60Hz
                "bass": np.mean(log_mel_spec[5:10, :]),           # 60-120Hz
                "low_mid": np.mean(log_mel_spec[10:20, :]),       # 120-250Hz
                "mid": np.mean(log_mel_spec[20:40, :]),           # 250-500Hz
                "high_mid": np.mean(log_mel_spec[40:80, :]),      # 500-2000Hz
                "presence": np.mean(log_mel_spec[80:120, :]),     # 2000-4000Hz
                "brilliance": np.mean(log_mel_spec[120:, :])      # 4000Hz~
            }
            
            # 楽器の特性に基づいて存在確率を計算
            # ベース: 低音域が強調
            if bands["sub_bass"] > -50 and bands["bass"] > -40:
                instruments["ベース"] = min(1.0, 0.5 + (bands["bass"] + 60) / 40)
            
            # ドラム: 過渡特性と広帯域ノイズ
            # 過渡的な変化を検出
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            if np.max(onset_strength) > 0.2:
                instruments["ドラム"] = min(1.0, np.max(onset_strength) * 1.5)
            
            # ギター: 中域と高中域の特性
            if bands["mid"] > -50 and bands["high_mid"] > -45:
                instruments["ギター"] = min(1.0, 0.3 + (bands["high_mid"] + 50) / 30)
            
            # ピアノ: 広い周波数帯域と特徴的な倍音構造
            piano_factor = (bands["low_mid"] + bands["mid"] + bands["high_mid"]) / 3
            if piano_factor > -45:
                instruments["ピアノ"] = min(1.0, 0.3 + (piano_factor + 50) / 20)
            
            # シンセ: 特徴的な周波数分布とサステイン
            synth_factor = bands["high_mid"] + bands["presence"]
            if synth_factor > -80:
                instruments["シンセサイザー"] = min(1.0, 0.2 + (synth_factor + 100) / 50)
            
            # ボーカル: 人声の特徴的な周波数帯域
            if bands["mid"] > -45 and bands["high_mid"] > -50:
                vocal_range = np.mean(log_mel_spec[30:70, :])  # 人声の主要周波数帯域
                instruments["ボーカル"] = min(1.0, 0.3 + (vocal_range + 60) / 30)
            
            # ダンス性の詳細計算
            debug_print("Calculating danceability...")
            danceability = calculate_danceability(y, sr)
            
            # アコースティック度の計算
            # 高周波ノイズの少なさ、倍音構造の自然さなどを考慮
            spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            # 低コントラストはアコースティック楽器の特徴
            contrast_mean = np.mean(spectral_contrast)
            
            # 高周波数成分の割合
            high_freq_energy = np.sum(S[int(S.shape[0]*0.6):, :]) / np.sum(S)
            
            # アコースティック度を計算
            acousticness = 1.0 - (0.5 * min(1.0, high_freq_energy * 10) + 0.5 * min(1.0, contrast_mean / 20))
            
            # セクション検出（拡張版）
            debug_print("Detecting sections with novelty detection...")
            
            # スペクトル特徴量の変化を検出
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 構造分析 (librosa.segment.onset_strength_multi が存在しないため代替手段を使用)
            # 複数のバンドでオンセット強度を計算
            try:
                n_bands = 6
                
                # librosa.filterbank が存在しない場合は、librosa.filters を使用
                from librosa.filters import mel as mel_filters
                mel_basis = mel_filters(sr=sr, n_fft=2048, n_mels=n_bands)
                
                S = np.abs(librosa.stft(y))
                novelty_curves = []
                
                # 各バンドごとにオンセット検出を実行
                for i in range(n_bands):
                    band_energy = np.sum(mel_basis[i].reshape(-1, 1) * S, axis=0)
                    # Use y and sr instead of directly passing onset_envelope
                    band_onset = librosa.onset.onset_strength(S=librosa.util.normalize(band_energy.reshape(1, -1)))
                    novelty_curves.append(band_onset)
            except Exception as e:
                debug_print(f"Error in multi-band onset detection: {e}")
                # 単一バンドでのオンセット検出にフォールバック
                novelty_curves = [librosa.onset.onset_strength(y=y, sr=sr)]
                
            # 全バンドのオンセット強度を合計
            novelty_curves = np.array(novelty_curves)
            novelty_sum = np.sum(novelty_curves, axis=0)
            
            # ピークを検出
            peaks = librosa.util.peak_pick(novelty_sum, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
            
            # セクション数をピーク数から推定
            estimated_sections = len(peaks) + 1
            # 現実的な範囲に制限
            sections = max(2, min(10, estimated_sections))
            
            return {
                "tempo": float(tempo),
                "key": key_result,
                "energy": float(energy),
                "danceability": float(danceability.item()) if isinstance(danceability, np.ndarray) else float(danceability),
                "acousticness": float(acousticness),
                "instruments": instruments,
                "sections": sections
            }
            
        except Exception as e:
            debug_print(f"Error during advanced audio analysis: {str(e)}")
            # エラー時はモックデータを返す
            return generate_mock_analysis()
    
    def analyze_music_file(file_path):
        """
        音楽ファイルの高度な分析を行う
        
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
            debug_print(f"Starting advanced analysis for: {file_path}")
            
            # 音声ファイルを読み込み
            y, sr = librosa.load(file_path, mono=True, duration=30)
            debug_print(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
            
            # 特徴量抽出
            features = extract_features(y, sr)
            
            # ジャンル分類
            genres = predict_genre(features)
            
            # 詳細な音楽分析
            analysis = analyze_audio(file_path, y, sr)
            
            # 音楽属性の検出
            attributes = detect_attributes(y, sr, analysis)
            
            # 波形データ生成
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
            description = generate_mock_description(genres, analysis, attributes)
            
            return {
                "genres": genres,
                "waveform": waveform,
                "analysis": analysis,
                "attributes": attributes,
                "description": description
            }
            
        except Exception as e:
            debug_print(f"Error during advanced music file analysis: {str(e)}")
            # エラー時はモックデータを返す
            return generate_mock_results()
    
def analyze_advanced_music(file_path):
    """
    音楽ファイルの高度な分析を行う（メインエントリーポイント）
    
    Parameters:
    -----------
    file_path : str
        音楽ファイルのパス
    
    Returns:
    --------
    results_json : str
        分析結果（JSON形式）
    """
    debug_print(f"Advanced analysis request received for: {file_path}")
    
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
        debug_print(f"Error in analyze_advanced_music: {str(e)}")
        # エラーオブジェクトを返す
        error_result = {
            "error": True,
            "message": f"高度な音楽分析中にエラーが発生しました: {str(e)}",
        }
        return json.dumps(error_result)

# スクリプトとして実行された場合
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='高度な音楽ジャンル分類と分析')
    parser.add_argument('file_path', help='分析する音楽ファイルのパス')
    parser.add_argument('analysis_id', nargs='?', default=None, help='分析ID')
    parser.add_argument('method', nargs='?', default='advanced', help='分析手法')
    parser.add_argument('--debug', action='store_true', help='デバッグ出力を有効にする')
    
    args = parser.parse_args()
    
    # デバッグモードの設定
    if args.debug:
        debug_print("デバッグモードが有効化されました")
    
    # ファイルの存在確認
    if not os.path.exists(args.file_path):
        print(f"エラー: ファイルが見つかりません: {args.file_path}")
        sys.exit(1)
    
    debug_print(f"Processing file: {args.file_path}")
    print(f"Mode: {'MOCK MODE' if USE_MOCK else 'NORMAL MODE'}", file=sys.stderr)
    print(f"Processing file: {args.file_path}", file=sys.stderr)
    print(f"Analysis ID: {args.analysis_id}", file=sys.stderr)  # Added debug output
    print(f"Method: {args.method}", file=sys.stderr)  # Added debug output
    results = analyze_advanced_music(args.file_path)
    
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
