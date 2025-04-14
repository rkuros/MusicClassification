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

# 共通のモック関数
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
