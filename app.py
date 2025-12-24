"""
FLUX.2 Gradio App
画像とテキストプロンプトを入力して、FLUX.2で画像生成を行うWebアプリ
"""

import gc

import gradio as gr
import torch
from diffusers import Flux2Pipeline
from PIL import Image


class FLUX2App:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.repo_id = "diffusers/FLUX.2-dev-bnb-4bit"

    def load_model(self):
        """モデルを初回のみロード"""
        if self.pipe is None:
            print("モデルをロード中...")
            self.pipe = Flux2Pipeline.from_pretrained(
                self.repo_id,
                torch_dtype=self.torch_dtype
            ).to(self.device)

            # メモリ最適化
            self.pipe.enable_model_cpu_offload()
            print("モデルのロード完了")
        return self.pipe

    def generate_image(
        self,
        prompt: str,
        input_image: Image.Image,
        num_steps: int = 28,
        guidance_scale: float = 4.0,
        seed: int = 42,
        width: int = 1024,
        height: int = 768,
        progress=gr.Progress()
    ):
        """
        画像を生成

        Args:
            prompt: テキストプロンプト
            input_image: 入力画像（任意）
            num_steps: デノイジングステップ数
            guidance_scale: ガイダンススケール
            seed: 乱数シード
            width: 出力画像の幅
            height: 出力画像の高さ
            progress: Gradio Progress tracker

        Returns:
            生成された画像、ステータスメッセージ
        """
        try:
            # プロンプトチェック
            if not prompt or prompt.strip() == "":
                error_msg = "❌ エラー: プロンプトを入力してください"
                progress(0, desc=error_msg)
                return None, error_msg

            # モデルロード
            progress(0.1, desc="🔄 モデルをロード中...")
            pipe = self.load_model()

            # 生成パラメータ
            progress(0.2, desc="⚙️ パラメータを設定中...")
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # 入力画像の処理
            images_input = [input_image] if input_image is not None else None

            # 画像生成
            progress(0.3, desc=f"🎨 画像生成中... (0/{num_steps} steps)")
            print(f"生成開始: prompt='{prompt[:50]}...', steps={num_steps}, guidance={guidance_scale}")

            # コールバック関数でプログレスを更新
            def callback(pipe, step_index, timestep, callback_kwargs):
                progress_value = 0.3 + (0.6 * (step_index + 1) / num_steps)
                progress(progress_value, desc=f"🎨 画像生成中... ({step_index + 1}/{num_steps} steps)")
                return callback_kwargs

            result = pipe(
                prompt=prompt,
                image=images_input,
                generator=generator,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                callback_on_step_end=callback,
            )

            output_image = result.images[0]

            # メモリクリア
            progress(0.95, desc="🧹 メモリをクリア中...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            success_msg = f"✅ 生成完了！ (steps={num_steps}, guidance={guidance_scale}, seed={seed})"
            progress(1.0, desc=success_msg)
            return output_image, success_msg

        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"❌ VRAM不足エラー: メモリが足りません。ステップ数や解像度を下げてください。\n詳細: {str(e)}"
            print(error_msg)
            progress(0, desc="❌ VRAM不足エラー")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            return None, error_msg
        except Exception as e:
            error_msg = f"❌ エラーが発生しました: {type(e).__name__}\n詳細: {str(e)}"
            print(error_msg)
            progress(0, desc=f"❌ {type(e).__name__}")
            return None, error_msg


def create_ui():
    """Gradio UIを作成"""
    app = FLUX2App()

    with gr.Blocks(title="FLUX.2 画像生成", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 FLUX.2 画像生成アプリ

            テキストプロンプトと入力画像（任意）から、FLUX.2で新しい画像を生成します。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 入力エリア
                gr.Markdown("### 入力")

                prompt_input = gr.Textbox(
                    label="プロンプト",
                    placeholder="生成したい画像の説明を入力してください（例: a beautiful sunset over the ocean）",
                    lines=3,
                    value="a beautiful sunset over the ocean with vibrant colors"
                )

                image_input = gr.Image(
                    label="入力画像（任意）",
                    type="pil",
                    sources=["upload", "clipboard"]
                )

                with gr.Accordion("詳細設定", open=False):
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=28,
                        step=1,
                        label="ステップ数（多いほど高品質だが時間がかかる）"
                    )

                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.5,
                        label="ガイダンススケール（高いほどプロンプトに忠実）"
                    )

                    seed = gr.Number(
                        label="シード値（再現性確保）",
                        value=42,
                        precision=0
                    )

                    with gr.Row():
                        width = gr.Slider(
                            minimum=512,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="幅"
                        )
                        height = gr.Slider(
                            minimum=512,
                            maximum=2048,
                            value=768,
                            step=64,
                            label="高さ"
                        )

                generate_btn = gr.Button("🎨 生成", variant="primary", size="lg")

            with gr.Column(scale=1):
                # 出力エリア
                gr.Markdown("### 出力")

                output_image = gr.Image(
                    label="生成画像",
                    type="pil"
                )

                status_text = gr.Textbox(
                    label="ステータス",
                    interactive=False
                )

        # サンプル例
        gr.Markdown("### 📝 サンプルプロンプト例")
        gr.Examples(
            examples=[
                ["a photo of a forest with mist swirling around the tree trunks"],
                ["a clean monochrome CAD-style technical line drawing"],
                ["a beautiful landscape with mountains and a lake at sunset"],
                ["an astronaut riding a horse on the moon"],
                ["a cute cat wearing sunglasses, digital art"],
            ],
            inputs=[prompt_input],
            label="クリックしてプロンプトをセット"
        )

        # イベント設定
        generate_btn.click(
            fn=app.generate_image,
            inputs=[
                prompt_input,
                image_input,
                num_steps,
                guidance_scale,
                seed,
                width,
                height,
            ],
            outputs=[output_image, status_text]
        )

        gr.Markdown(
            """
            ---
            **使い方:**
            1. プロンプトを入力（必須）
            2. 入力画像をアップロード（任意、編集モードの場合）
            3. 詳細設定を調整（任意）
            4. 「生成」ボタンをクリック

            **注意:**
            - 初回実行時はモデルのロードに時間がかかります
            - VRAM不足の場合はステップ数や解像度を下げてください
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
