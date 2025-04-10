# MusicClassification - 音楽ジャンル分類 Webアプリケーション

MP3音楽ファイルをアップロードすると、機械学習を使用して音楽のジャンルを分析・分類するWebアプリケーションです。

## 特徴

- ドラッグ＆ドロップによる簡単なMP3ファイルアップロード
- 高精度な音楽ジャンル分析（rock, pop, jazz, classical, electronic, hip-hop, country, blues, reggae, metalなど）
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

- **フロントエンド**: HTML5のドラッグ＆ドロップAPIとFetch APIを使用したシンプルなUI
- **バックエンド**: Express.jsによるRESTful API
- **データベース**: MongoDBによる分析結果の永続化
- **音楽分析**: 
  - Librosaによる音楽特徴量抽出（MFCC, スペクトル特性, テンポ, リズム特性など）
  - Scikit-learnの機械学習モデルによるジャンル分類

## ライセンス

MIT
