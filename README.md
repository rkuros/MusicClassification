# MusicClassification - 音楽ジャンル分類 Webアプリケーション

MP3音楽ファイルをアップロードすると、機械学習を使用して音楽のジャンルを分析・分類するWebアプリケーションです。

## 特徴

- ドラッグ＆ドロップによる簡単なMP3ファイルアップロード
- 高精度な音楽ジャンル分析（rock, pop, jazz, classical, electronic, hip-hop, country, blues, reggae, metalなど）
- **波形の視覚的表示**（アップロードした音楽のオーディオ波形をCanvas APIで表示）
- **詳細な音楽分析情報**（テンポ、キー、エネルギー、ダンス性、アコースティック度など）
- **楽器構成の分析**（ピアノ、ギター、ドラム、ベース、シンセサイザー、ボーカルなど）
- **音楽の総合的な説明文生成**（ジャンルと音楽特性に基づく自然言語による説明）
- 分析結果の視覚的表示（ジャンルごとの信頼度スコア）
- 分析履歴の保存と表示機能
- レスポンシブデザインに対応

## システム要件

### フロントエンド
- HTML5, CSS3, JavaScript

### バックエンド
- Node.js v14.0以上
- Python 3.8以上
- MongoDB

### 必須ライブラリ
- Node.js: Express, Multer, Mongoose, UUID, CORS
- Python: Librosa, NumPy, Scikit-learn, Joblib

## インストール方法

1. リポジトリをクローン
```
git clone https://github.com/rkuros/MusicClassification.git
cd MusicClassification
```

2. 依存関係のインストール
```
npm run install-deps
```

3. MongoDBの起動
```
# MongoDBがローカルにインストールされている場合
mongod
```

## 使用方法

1. アプリケーションの起動
```
npm start
```

2. ブラウザでアクセス
```
http://localhost:3000
```

3. MP3ファイルをアップロード
   - ファイルをドラッグ＆ドロップするか、「ファイルを選択」ボタンをクリック

4. 分析結果の確認
   - アップロード後、自動的に分析が行われます
   - 分析結果は信頼度の高い順にジャンルが表示されます

## プロジェクト構成

```
MusicClassification/
├── public/              # フロントエンドファイル
│   ├── index.html       # メインHTML
│   ├── styles.css       # スタイルシート
│   └── app.js           # フロントエンドJavaScript
├── server/              # バックエンドサーバー
│   ├── server.js        # Express サーバー
│   └── uploads/         # アップロードされたファイルの保存先
├── python/              # 音楽分析スクリプト
│   ├── genre_classifier.py  # ジャンル分類スクリプト
│   ├── requirements.txt     # Pythonの依存関係
│   ├── model.pkl            # 学習済みモデル（自動生成）
│   └── scaler.pkl           # 特徴量スケーラー（自動生成）
├── package.json         # プロジェクト情報とNode.js依存関係
└── README.md            # このファイル
```

## 技術詳細

- **フロントエンド**: 
  - HTML5のドラッグ＆ドロップAPIとFetch APIを使用したシンプルなUI
  - Canvas APIを使った音楽波形の視覚的表示
  - グリッドレイアウトを活用した詳細分析情報の表示
  - レスポンシブデザインによるモバイル対応

- **バックエンド**: 
  - Express.jsによるRESTful API
  - 標準出力バッファリングによるPythonスクリプト実行結果の安定した処理
  - 非同期処理による長時間の分析タスク管理

- **データベース**: 
  - MongoDBによる分析結果の永続化
  - 拡張スキーマによる波形データ、詳細分析情報、説明文の格納

- **音楽分析**: 
  - Librosaライブラリによる高度な音楽特徴量抽出
    - MFCC（メル周波数ケプストラム係数）による音色特性分析
    - スペクトル特性（重心、帯域幅、ロールオフ等）の算出
    - テンポ、リズム解析、キー推定
    - ゼロ交差率、RMSエネルギー計算
  - 波形データのダウンサンプリングと視覚化処理
  - 楽器構成の検出アルゴリズム
    - スペクトル特性に基づく楽器タイプの推定
    - ハーモニック・パーカッシブ分離による楽器検出
  - 自然言語による音楽説明文の自動生成
  - Scikit-learnの機械学習モデル（RandomForest）によるジャンル分類
  - NumPy 2.0互換性対応

- **データ処理**:
  - 大規模データ（波形等）のJSON変換処理の最適化
  - NumPy配列データを標準Pythonデータ型に変換するヘルパー関数
  - バッファリングによる分割データの安全な結合処理

## ライセンス

MIT
