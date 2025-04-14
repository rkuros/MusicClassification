#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import json

def test_classifier(audio_file_path):
    """
    Test the advanced classifier directly and print the results in a structured way
    to verify the output format and content.
    
    Parameters:
    -----------
    audio_file_path : str
        Path to the MP3 file for testing
    """
    if not os.path.exists(audio_file_path):
        print(f"エラー: テスト用音声ファイルが見つかりません: {audio_file_path}")
        return False
        
    print(f"テスト: {audio_file_path} の分析を開始します")
    
    # advanced_classifier.py を実行
    script_path = os.path.join(os.path.dirname(__file__), 'advanced_classifier.py')
    
    # サブプロセスを実行し、標準出力と標準エラー出力を分けて取得
    process = subprocess.Popen(
        ['python3', script_path, audio_file_path, 'test_id', 'advanced'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 出力を取得
    stdout, stderr = process.communicate()
    
    # 標準エラー出力を表示（デバッグ情報）
    print("\n===== デバッグ情報（stderr） =====")
    print(stderr)
    
    # 標準出力（JSON）をパース
    print("\n===== JSON出力（stdout） =====")
    print(f"JSON出力の長さ: {len(stdout)} 文字")
    
    try:
        # JSON解析テスト
        result = json.loads(stdout)
        print("JSON解析成功!")
        
        # 基本的なキーの存在確認
        expected_keys = ['genres', 'waveform', 'analysis', 'description']
        missing_keys = [key for key in expected_keys if key not in result]
        
        if missing_keys:
            print(f"警告: 次のキーがJSONに見つかりません: {', '.join(missing_keys)}")
        else:
            print("すべての必須キーが存在します")
            
            # ジャンル情報の確認
            print(f"\nジャンル分類結果:")
            for genre in result['genres'][:3]:  # 上位3つのみ表示
                print(f"  - {genre['name']}: {genre['confidence']:.2f}")
            
            # 分析情報の一部を表示
            print(f"\n分析結果:")
            print(f"  - テンポ: {result['analysis']['tempo']:.1f} BPM")
            print(f"  - キー: {result['analysis']['key']}")
            print(f"  - エネルギー: {result['analysis']['energy']:.2f}")
            print(f"  - ダンス性: {result['analysis']['danceability']:.2f}")
            
            print("\nテスト完了: 出力フォーマットは正常です")
            return True
    except json.JSONDecodeError as e:
        print(f"エラー: JSON解析に失敗しました: {e}")
        print(f"JSON解析に失敗した出力の先頭部分:")
        print(stdout[:200] + "..." if len(stdout) > 200 else stdout)
        return False

if __name__ == "__main__":
    # コマンドライン引数からオーディオファイルのパスを取得
    if len(sys.argv) < 2:
        print("使用方法: python test_advanced_classifier.py <音声ファイルのパス>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    success = test_classifier(audio_path)
    
    if success:
        print("\nテスト成功: 分類器は正常に動作しています。")
    else:
        print("\nテスト失敗: 分類器の出力に問題があります。")
