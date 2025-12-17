# FLUX.2 Inference

FLUX.2 のオープンウェイトモデルをローカルで実行するための推論コードです。画像生成・編集に対応し、テキストエンコードから画像生成まで完全にローカルで処理できます。

by Black Forest Labs: https://bfl.ai
API ドキュメント: https://docs.bfl.ai


## 概要

- **完全ローカル実行**: テキストエンコーダー（Mistral3 Small）、フローモデル、オートエンコーダーをすべてローカルGPUで実行
- **プロンプトアップサンプリング**: ローカルまたはOpenRouter経由でプロンプトを拡張（任意）
- **モデル自動取得**: Hugging Face Hub から `black-forest-labs/FLUX.2-dev` を自動ダウンロード、または手元の `weights/` を利用
- **柔軟な実行方法**: CLI（対話モード・単発実行）、Diffusers パイプライン、Jupyter Notebook


## 動作要件

- **Python**: 3.10以上、3.13未満（推奨: 3.11）
- **GPU**: CUDA対応GPU（推奨）
  - VRAM: 約20GB以上（4-bit量子化構成）
  - VRAM不足時: `--cpu_offloading=True` でCPUへ一部オフロード可能
- **OS**: Linux / macOS / Windows（WSL2推奨）


## 環境構築

### uvによるセットアップ（推奨）

```bash
# uvのインストール（未インストールの場合）
pip install uv
# 依存関係のインストール
uv sync

# 仮想環境の有効化
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### Hugging Face ログイン

モデルを自動取得する場合、Hugging Face へのログインが必要です:

```bash
# gated モデルへのアクセス承認を事前に取得
# https://huggingface.co/black-forest-labs/FLUX.2-dev

uv run hf auth login
```


## モデルウェイトの指定（任意）

同梱の `weights/` ディレクトリを利用する場合、環境変数で指定できます:

```bash
export FLUX2_MODEL_PATH="$(pwd)/weights/flux2-dev.safetensors"
export AE_MODEL_PATH="$(pwd)/weights/ae.safetensors"
```

自動ダウンロード時のキャッシュ場所:
```
~/.cache/huggingface/hub/
```

カスタムキャッシュディレクトリを使う場合:
```bash
export HF_HOME=/path/to/custom/cache
```


## デモの利用方法
詳細は[Flux2 README](./README_origin.md)を参照。
### Diffusers パイプライン（Python）

ローカルでテキストエンコードを実行する例:
GPUのVRAMが24~32GBの場合は、以下の4bit量子化したモデルで完全にローカル環境で実行可能。

```python
import torch
from diffusers import Flux2Pipeline
from PIL import Image

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

# パイプラインの初期化
pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype
).to(device)

# メモリ最適化（任意）
pipe.enable_model_cpu_offload()

# 画像生成
prompt = "a photo of a forest with mist swirling around the tree trunks"
input_image = Image.open("input.jpg")  # 任意

image = pipe(
    prompt=prompt,
    image=[input_image],  # 省略可能
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=28,
    guidance_scale=4.0,
).images[0]

image.save("output.png")
```


### 4. Jupyter Notebook

プロジェクト内の `test.ipynb` にサンプルコードがあります:

```bash
uv run jupyter notebook test.ipynb
```


## パラメータ説明

| パラメータ             | 説明                                                  | デフォルト |
| ---------------------- | ----------------------------------------------------- | ---------- |
| `prompt`               | 生成する画像の説明文                                  | -          |
| `width` / `height`     | 出力画像サイズ（ピクセル）                            | 1360 / 768 |
| `num_steps`            | デノイジングステップ数                                | 50         |
| `guidance`             | ガイダンススケール（高いほどプロンプトに忠実）        | 4.0        |
| `seed`                 | 乱数シード（再現性確保）                              | ランダム   |
| `input_images`         | 入力画像パス（編集モード）                            | なし       |
| `match_image_size`     | 入力画像のサイズに合わせる（インデックス指定）        | なし       |
| `upsample_prompt_mode` | プロンプト拡張モード: `none` / `local` / `openrouter` | `none`     |
| `cpu_offloading`       | CPUオフロード有効化                                   | False      |




## 参考ドキュメント

- [Diffusers経由の利用方法](docs/flux2_dev_hf.md)
- [プロンプトアップサンプリング解説](docs/flux2_with_prompt_upsampling.md)
- [オリジナルREADME](README_origin.md)（存在する場合）
