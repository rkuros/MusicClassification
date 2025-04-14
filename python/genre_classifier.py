#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
from datetime import datetime

# 共通ユーティリティをインポート
from utils import debug_print

# スクリプト開始時のデバッグ情報
debug_print(f"Genre Classifier (Dispatcher) started at {datetime.now()}")

def main():
    """
    メイン関数: 分析手法に基づいて適切なクラシファイアにリクエストを振り分ける
    """
    # コマンドライン引数の処理
    if len(sys.argv) < 2:
        print("使用方法: python genre_classifier.py <音楽ファイルパス> [分析ID] [分析手法]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analysis_id = sys.argv[2] if len(sys.argv) > 2 else None
    method = sys.argv[3] if len(sys.argv) > 3 else "standard"
    
    debug_print(f"Dispatching request - File: {file_path}, ID: {analysis_id}, Method: {method}")
    
    # ファイルの存在確認
    if not os.path.exists(file_path):
        error_result = {
            "error": True,
            "message": f"エラー: ファイルが見つかりません: {file_path}"
        }
        print(json.dumps(error_result))
        sys.exit(1)
    
    # 分析手法に基づいてクラシファイアを選択
    classifier_script = ""
    if method.lower() == "advanced":
        classifier_script = os.path.join(os.path.dirname(__file__), "advanced_classifier.py")
        debug_print(f"Using advanced classifier: {classifier_script}")
    else:
        classifier_script = os.path.join(os.path.dirname(__file__), "standard_classifier.py")
        debug_print(f"Using standard classifier: {classifier_script}")
    
    # 選択したクラシファイアを実行
    try:
        # 分析IDを含む引数を構築
        command = [sys.executable, classifier_script, file_path]
        if analysis_id:
            command.append(analysis_id)
        
        debug_print(f"Executing: {' '.join(command)}")
        
        # サブプロセスとして実行
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 結果を取得
        stdout, stderr = process.communicate()
        
        # エラー出力を表示（デバッグ目的）
        if stderr:
            debug_print(f"Classifier stderr: {stderr}")
        
        # 標準出力を返す（JSONレスポンス）
        print(stdout)
        
        # 終了コードの確認
        if process.returncode != 0:
            debug_print(f"Classifier exited with code: {process.returncode}")
    
    except Exception as e:
        # エラーが発生した場合はJSONエラーオブジェクトを返す
        error_result = {
            "error": True,
            "message": f"分類処理中にエラーが発生しました: {str(e)}"
        }
        print(json.dumps(error_result))

# スクリプトとして実行された場合
if __name__ == "__main__":
    main()
