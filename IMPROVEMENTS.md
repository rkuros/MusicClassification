# 音楽ジャンル分類の改善点

このドキュメントでは、音楽ジャンル分類システムに対して行った改善点について説明します。

## 主な改善点

### 1. ジャンル分類の拡張

- **ジャンル一覧の拡充**: 従来の10種類のジャンルから16種類に拡大
  - 新規追加ジャンル: dance, funk, soul, folk, ambient, indie
  - これにより「Unnatural Attraction.mp3」のような曲を正確に「dance」として分類可能に

### 2. 音楽属性の検出

- **属性分類機能**: ジャンル以外の音楽特性を検出する機能を追加
  - 属性一覧: high energy, chill, acoustic, instrumental, vocal, fast tempo, slow tempo, upbeat, melancholic
  - 「high energy」などの音楽の雰囲気や特徴をより正確に表現可能に

### 3. 特徴量抽出の強化

- **分析時間の延長**: 30秒から60秒に延長し、より多くの音楽パターンを分析
- **MFCCの拡張**: 13次元から20次元に拡張し、分散も追加
- **クロマグラム分析の強化**: 平均だけでなく分散も特徴量として追加
- **スペクトル特徴量の追加**: スペクトルコントラスト、平坦度などを追加
- **音楽構造分析**: ハーモニックとパーカッシブ成分の分離解析
- **ダンスビリティ分析**: パルス強度の計測とダンスビリティの計算

### 4. 機械学習モデルの高度化

- **深層学習の導入**: PyTorchによる多層ニューラルネットワークモデルを追加
  - 従来のRandomForestに加えて、より高度な学習が可能に
  - ドロップアウト層を使った過学習の防止
- **パラメーターの最適化**: RandomForestのパラメーターを最適化（木の数、深さなど）

### 5. UI/UXの改善

- **属性表示の追加**: ジャンルだけでなく、音楽属性も表示するように改善
- **レイアウトの最適化**: 結果表示画面のレイアウトを改善し、情報を整理して表示

## 技術的詳細

### 特徴量抽出の詳細

特徴量抽出プロセスは次の要素を含みます：

1. **テンポラル特徴量**: 時間的変化を捉える特徴量
   - ゼロ交差率とその分散
   - ビート強度

2. **スペクトル特徴量**: 周波数特性を捉える特徴量
   - スペクトル重心、帯域幅
   - スペクトル平坦度、ロールオフ

3. **エネルギー特徴量**: 音量とエネルギー分布を表す特徴量
   - RMSエネルギー
   - 総エネルギー

4. **構造的特徴量**: 音楽の構造を表す特徴量
   - ハーモニック成分
   - パーカッシブ成分

### 深層学習モデルのアーキテクチャ

```
Input Layer (特徴量の次元数) → 
  Dense Layer (128 units) + ReLU + Dropout(0.3) → 
    Dense Layer (64 units) + ReLU + Dropout(0.2) → 
      Output Layer (ジャンル数) + Softmax
```

## 今後の改善可能性

1. **実データによる訓練**: 実際の音楽データセット（GTZAN, FMAなど）を用いたモデル訓練
2. **畳み込みニューラルネットワーク(CNN)の導入**: スペクトログラムを直接入力として使用
3. **転移学習の活用**: VGGishやPANNsなどの事前学習済みモデルの活用
4. **ユーザーフィードバックの統合**: 継続的な学習のためのフィードバックシステムの構築

## 依存関係

改善されたシステムでは、次のライブラリを使用しています：

- librosa: 音声分析
- numpy: 数値計算
- scikit-learn: 機械学習
- PyTorch: 深層学習
- torchaudio: 音声処理用ライブラリ
