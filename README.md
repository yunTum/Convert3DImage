# 画像から奥行き推定アプリ

このアプリケーションは、ユーザーがアップロードした画像から奥行き（深度）マップを生成します。深層学習モデルを使用して、2D画像から3D深度情報を推定します。

## 機能

- 画像のアップロード
- 深度マップの生成と表示
- 深度マップのダウンロード

## 技術スタック

- Streamlit: UIフレームワーク
- PyTorch: 深層学習フレームワーク
- Transformers: Hugging Faceの事前学習済みモデル（DPT）
- Pillow: 画像処理
- NumPy: 数値計算

## インストールと実行方法

1. 必要なパッケージをインストール:
```
pip install -r requirements.txt
```

2. アプリケーションを実行:
```
streamlit run app.py
```

3. ブラウザで `http://localhost:8501` にアクセス

## 使用しているモデル

このアプリはIntelのDepth Prediction Transformer (DPT) モデルを使用しています。 