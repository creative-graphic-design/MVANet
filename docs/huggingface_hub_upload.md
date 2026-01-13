# Hugging Face Hub へのアップロードガイド

このドキュメントでは、MVANet モデルを Hugging Face Hub にアップロードする方法を説明します。

## 前提条件

### 1. Hugging Face アカウントの作成

Hugging Face Hub にモデルをアップロードするには、アカウントが必要です：

- [Hugging Face](https://huggingface.co/join) でアカウントを作成

### 2. アクセストークンの取得

1. [Settings > Access Tokens](https://huggingface.co/settings/tokens) にアクセス
2. "New token" をクリック
3. Token の名前を入力（例: "mvanet-upload"）
4. Role として "Write" を選択
5. "Generate a token" をクリック
6. 生成されたトークンをコピー（後で使用します）

### 3. 必要なパッケージのインストール

```bash
pip install huggingface_hub
```

または、プロジェクトのすべての依存関係をインストール：

```bash
uv sync
```

### 4. Hugging Face CLI でログイン

```bash
huggingface-cli login
```

プロンプトが表示されたら、手順 2 で取得したトークンを入力します。

## アップロード方法

### 基本的な使い方

```bash
python scripts/push_to_hub.py --repo-id your-username/mvanet
```

### オプション

- `--repo-id`: **必須** - Hugging Face Hub のリポジトリ ID（例: `creative-graphic-design/mvanet`）
- `--token`: オプション - Hugging Face API トークン（ログイン済みの場合は不要）
- `--private`: オプション - プライベートリポジトリとして作成
- `--local-dir`: オプション - モデルを保存するローカルディレクトリ（デフォルト: `./hf_model`）
- `--model-card`: オプション - モデルカードのパス（デフォルト: `MODEL_CARD.md`）

### 例

#### 1. パブリックリポジトリとして公開

```bash
python scripts/push_to_hub.py --repo-id creative-graphic-design/mvanet
```

#### 2. プライベートリポジトリとして作成

```bash
python scripts/push_to_hub.py --repo-id your-username/mvanet-private --private
```

#### 3. トークンを直接指定

```bash
python scripts/push_to_hub.py --repo-id your-username/mvanet --token hf_xxxxxxxxxxxxx
```

#### 4. カスタムローカルディレクトリを使用

```bash
python scripts/push_to_hub.py --repo-id your-username/mvanet --local-dir ./my_model_dir
```

## アップロードプロセス

スクリプトは以下の手順を実行します：

1. **モデルの読み込み**: 訓練済み MVANet モデルの重みを読み込み
2. **Transformers 形式への変換**: Hugging Face Transformers と互換性のある形式に変換
3. **ローカル保存**: モデル、設定、プロセッサーをローカルディレクトリに保存
4. **モデルカードのコピー**: `MODEL_CARD.md` を `README.md` としてコピー
5. **リポジトリ作成**: Hugging Face Hub にリポジトリを作成（存在しない場合）
6. **アップロード**: すべてのファイルを Hugging Face Hub にアップロード

## アップロード内容

以下のファイルがアップロードされます：

```
your-username/mvanet/
├── README.md                    # モデルカード（MODEL_CARD.md のコピー）
├── config.json                  # モデル設定
├── preprocessor_config.json     # プロセッサー設定
└── pytorch_model.bin            # モデルの重み
```

## トラブルシューティング

### エラー: "Authentication required"

```bash
huggingface-cli login
```

を実行してログインしてください。

### エラー: "Repository not found"

リポジトリ ID が正しいか確認してください。形式は `username/model-name` です。

### エラー: "Permission denied"

- アクセストークンが "Write" 権限を持っているか確認
- リポジトリが既に存在する場合、そのリポジトリへの書き込み権限があるか確認

### アップロードが遅い

モデルサイズは約 500MB あります。ネットワーク速度によっては数分かかる場合があります。

## アップロード後の確認

アップロードが完了すると、以下の URL でモデルを確認できます：

```
https://huggingface.co/your-username/mvanet
```

## モデルの使用

アップロード後、以下のコードでモデルを使用できます：

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

# モデルとプロセッサーの読み込み
model = AutoModel.from_pretrained("your-username/mvanet")
processor = AutoImageProcessor.from_pretrained("your-username/mvanet")

# 画像の読み込みと推論
image = Image.open("image.jpg")
inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
masks = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)
```

## 注意事項

1. **ライセンス**: モデルカードに記載されているライセンス（MIT）を確認してください
2. **モデルサイズ**: 約 500MB のモデルをアップロードするため、十分なストレージがあるか確認
3. **Hugging Face の利用規約**: [Hugging Face Terms of Service](https://huggingface.co/terms-of-service) に従ってください

## 参考資料

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Uploading Models Guide](https://huggingface.co/docs/hub/models-uploading)
- [Model Cards Guide](https://huggingface.co/docs/hub/model-cards)

## サポート

問題が発生した場合は、[GitHub Issues](https://github.com/creative-graphic-design/MVANet/issues) で報告してください。
