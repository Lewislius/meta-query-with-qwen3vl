#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for Qwen3-VL by the project contributors.
#
# Gradio Web Demo for MetaQuery + Qwen3-VL
# Supports text-to-image and instruction-guided image generation.

import gradio as gr
import numpy as np
import torch

from pipeline_metaquery import MetaQueryPipeline
from trainer_utils import find_newest_checkpoint
import random
import argparse

MIN_SEED = 0
MAX_SEED = np.iinfo(np.int32).max
MAX_INPUT_IMAGES = 4
DEFAULT_INPUT_IMAGES = 1
MAX_IMAGES_PER_PROMPT = 4
DEFAULT_IMAGES_PER_PROMPT = 1

PRESET_NEGATIVE_PROMPTS = {
    "无": "",
    "基础": "low resolution, low quality, blurry",
    "详细": "bad anatomy, signature, watermark, username, error, missing limbs, error",
    "艺术风格": "photographic, realistic, photo-realistic, sharp focus, 3d render, oversaturated",
}


def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def variable_images(k):
    k = int(k)
    return [gr.update(visible=True)] * k + [gr.update(visible=False)] * (
        MAX_INPUT_IMAGES - k
    )


def process_interleaved_vision_language(
    prompt,
    negative_prompt,
    seed,
    guidance_scale,
    image_guidance_scale,
    num_inference_steps,
    num_images_per_prompt,
    *input_images,
):
    valid_images = [img for img in input_images if img is not None]

    images = pipeline(
        image=[valid_images] if len(valid_images) > 0 else None,
        caption=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=torch.Generator().manual_seed(seed),
        enable_progress_bar=True,
    ).images
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaQuery + Qwen3-VL Web Demo")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the trained checkpoint directory"
    )
    parser.add_argument(
        "--share", action="store_true", default=False,
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--server_port", type=int, default=7860,
        help="Server port"
    )
    args = parser.parse_args()

    print(f"Loading MetaQuery + Qwen3-VL pipeline from: {args.checkpoint_path}")
    pipeline = MetaQueryPipeline.from_pretrained(
        find_newest_checkpoint(args.checkpoint_path),
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    print("Pipeline loaded successfully!")

    with gr.Blocks(
        fill_width=True,
        title="MetaQuery + Qwen3-VL",
    ) as demo:
        gr.Markdown(
            """
            # 🎨 MetaQuery + Qwen3-VL
            **基于 Qwen3-VL 的 MetaQuery 图像生成系统**

            支持以下模式：
            - **文本生成图像 (T2I)**：输入文本描述，生成对应图像
            - **图像+文本生成图像 (I2I)**：输入参考图和文本指令，生成新图像
            """
        )

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="提示词 (Prompt)",
                    max_lines=3,
                    placeholder="描述你想生成的图像...",
                )
                negative_prompt_preset = gr.Dropdown(
                    choices=list(PRESET_NEGATIVE_PROMPTS.keys()),
                    value="无",
                    label="负面提示词预设",
                )
                negative_prompt = gr.Textbox(
                    label="负面提示词 (Negative Prompt)",
                    max_lines=1,
                    value="",
                )

                def update_negative_prompt(preset_name):
                    return PRESET_NEGATIVE_PROMPTS[preset_name]

                negative_prompt_preset.change(
                    fn=update_negative_prompt,
                    inputs=[negative_prompt_preset],
                    outputs=[negative_prompt],
                )

                seed = gr.Slider(
                    label="随机种子", minimum=MIN_SEED, maximum=MAX_SEED, step=1, value=0
                )
                randomize_seed = gr.Checkbox(label="随机化种子", value=True)
                guidance_scale = gr.Slider(
                    1, 30, step=0.5, value=4.5, label="文本引导强度 (Guidance Scale)"
                )
                image_guidance_scale = gr.Slider(
                    1, 30, step=0.5, value=1.5, label="图像引导强度 (Image Guidance Scale)"
                )
                with gr.Accordion("高级选项", open=False):
                    num_inference_steps = gr.Slider(
                        1, 100, step=1, value=30, label="推理步数"
                    )
                    num_images_per_prompt = gr.Slider(
                        1,
                        MAX_IMAGES_PER_PROMPT,
                        value=DEFAULT_IMAGES_PER_PROMPT,
                        step=1,
                        label="每次生成图片数量",
                    )
                generate_btn = gr.Button("🚀 生成图像", variant="primary")

                gr.Markdown("### 参考图像（可选）")
                num_input_images = gr.Slider(
                    1,
                    MAX_INPUT_IMAGES,
                    value=DEFAULT_INPUT_IMAGES,
                    step=1,
                    label="参考图像数量",
                )
                input_images = [
                    gr.Image(
                        label=f"参考图 {i+1}",
                        type="pil",
                        visible=True if i < DEFAULT_INPUT_IMAGES else False,
                    )
                    for i in range(MAX_INPUT_IMAGES)
                ]
                num_input_images.change(variable_images, num_input_images, input_images)

            with gr.Column():
                output_gallery = gr.Gallery(
                    columns=2, label="生成结果", show_label=True
                )

        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process_interleaved_vision_language,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                guidance_scale,
                image_guidance_scale,
                num_inference_steps,
                num_images_per_prompt,
                *input_images,
            ],
            queue=False,
            outputs=output_gallery,
        )

        generate_btn.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process_interleaved_vision_language,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                guidance_scale,
                image_guidance_scale,
                num_inference_steps,
                num_images_per_prompt,
                *input_images,
            ],
            queue=False,
            outputs=output_gallery,
        )

        demo.launch(
            share=args.share,
            server_port=args.server_port,
        )
