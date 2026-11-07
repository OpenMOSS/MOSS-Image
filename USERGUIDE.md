# MOSS-Image-Base User Guide

## Inference Parameters

As a base model, MOSS-Image-Base has not undergone any post-training or distillation of inference parameters such as CFG. Our goal is to preserve the training information in full and make it easier for the community to train LoRAs. This also means that users need suitable inference settings to get the best results. Here are our recommendations:

* **Resolution**: We recommend outputs with **1M, 1.5M, or 2M** total pixels. The model can also extrapolate to arbitrary resolutions from 0.25M (`512×512`) to 4M (2K).
* **Aspect Ratio**: We support arbitrary aspect ratios. Gradio’s resolution and aspect-ratio controls calculate dimensions aligned to multiples of 16. In the Python API, or when entering dimensions manually, use positive multiples of 16; invalid dimensions raise an error.
* **Inference Steps**: Best results are achieved with **≥ 50** steps.
* **CFG**: Default **3.0** (recommended range: 3.0–6.0). A value of 3.0 is generally sufficient for long prompts; for short prompts, we recommend increasing it as needed.
* **Shift**: Default **5.0**.
* **Sampler**: **Euler** is the recommended default.
* **SLG (Skip Layer Guidance)**: We have implemented two variants:
  * **Standard SLG**: scale 3, 0.01-1.0.
  * **Two-Pass Sampling SLG**: We recommend `uncond skip`, with **22** as the recommended skip layer (0-indexed). Layers 22, 24, and 26 can all help improve image quality.
  
  Turning off SLG will make the output image look very realistic, but it may also cause some details to collapse.

## Prompt Guidelines

MOSS-Image-Base supports multiple input formats, including natural language, tag sequences, and JSON layouts.

* **Language**: We recommend **English**, which has the most training data. Chinese and other languages are also supported.
* **Negative Prompt**: Based on our tests, we recommend the following effective negative tags:
  ```text
  nsfw, worst quality, bad anatomy, watermark, jpeg artifact
  ```
* **Recommended Tag Order**: For the best results with tag sequences, we recommend the order Quality -> Score -> Meta -> Year -> Character -> Series -> Artist -> General. For example:
  ```text
  best quality, highres, hatsune miku, vocaloid, @foo bar, 1girl, blue hair
  ```
* **Text Layout**: The model can render text and insert line breaks automatically, but the result may not always match your expectations. To improve text accuracy or control line breaks precisely, insert `\n` or actual line breaks directly into the prompt.

Here are some recommended example prompts:

```text
prompt:
一个穿着"MOSI"标志的T恤的中国美女正拿着黑色的马克笔面向镜头微笑。她身后的玻璃板上手写体写着 "一、MOSS-Image的技术路线：5B 的单流 DiT 模型，基于 Muon 训练。二、MOSS-Image的模型特色：1、复杂文字渲染。支持中英渲染、自动布局； 2、使用 Layout 进行精准图像控制，支持文字编辑和物体增减等自由编辑。三、MOSS-Image的后续发展：探索 R2I 参考生图、助力生成式AI发展。"

negative prompt:
worst quality, bad anatomy
```

```text
prompt:
A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
“记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。”

negative prompt:

```

### Layout Prompts (Experimental)

MOSS-Image-Base explores layout control for text-to-image generation and currently supports the following model-facing JSON format:

```json
{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top."
  },
  "items": [
    {
      "label": "text",
      "caption": "Large dark-blue uppercase title centered at the top.",
      "text": "EXPLORE",
      "bbox": [-416, -384, -288, 384]
    },
    {
      "label": "object",
      "caption": "A flat blue mountain illustration in the lower center.",
      "bbox": [-192, -288, 384, 288]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

The fields are defined as follows:

- `summary.caption`: A description of the overall scene and style.
- `items`: Each element's category (`label`), description (`caption`), and bounding box (`bbox`). Use `text` to specify the content when rendering text.
- `bbox_format: "yxyx"`: Coordinates are ordered as `[y_min, x_min, y_max, x_max]`.
- `bbox_coord_space: "abs_centralized"`: Absolute pixel coordinates with the canvas center as the origin (x increases to the right and y increases downward). For example, a 1024×1024 canvas ranges from -512 to +512 on both axes. For a canvas of width W and height H, subtract W/2 and H/2 from conventional top-left-origin x and y coordinates, respectively, to convert to this coordinate system.

> Some recent work uses relative coordinates for layout control. We use absolute pixel coordinates, which we believe are better suited to layout control across different resolutions and aspect ratios.

The model supports combining natural language and structured input.

#### Authoring editable layout drafts

The [Pixel Layout draft reference](skills/moss-image-prompt-enhancer/references/layout.md)
and [compiler](skills/moss-image-prompt-enhancer/scripts/compile_prompt.py) provide
an editable input format that differs from the model-facing JSON above:

| Format | Coordinate space | Box order | Canvas dimensions |
| --- | --- | --- | --- |
| Editable draft | `abs`, from the top-left | `xyxy` | `width` and `height` required |
| Model prompt | `abs_centralized`, from the center | `yxyx` | Omitted from the compiled JSON |

From the repository root, compile a draft with:

```bash
python3 skills/moss-image-prompt-enhancer/scripts/compile_prompt.py draft.json --output prompt.txt
```

For example, on a 1536×1024 canvas, the draft box `[176,224,800,864]`
becomes `[-288,-592,352,32]` in the model prompt. The compiler encloses
fractional draft edges at pixel boundaries before subtracting the canvas center.

Use the draft's width and height in your image-generation settings. Changing
these dimensions alone does not rescale the draft boxes; update their coordinates
when resizing a composition. Keep the editable draft for later revisions.

## Prompt Enhancer and Layout Enhancer

The MOSS-Image series natively supports layouts, enabling many interesting applications and expanding the models' capabilities. This also makes the models more difficult to use.

To address this, we use two approaches to layout enhancement. The training-free approach uses a carefully tuned skill with an LLM to provide basic layout planning. The other approach fine-tunes an open-source LLM directly as a layout enhancer. Stay tuned.

## Post-Training

### LoRA Training Recommendations

We have found that simple SFT can enable features such as GIF generation and reference-based generation in MOSS-Image-Base. Stay tuned.

If you post-train MOSS-Image-Base yourself, keep the text encoder frozen. Do not fine-tune it.

### Image Captioning

If you post-train MOSS-Image-Base yourself, you can use the following image captioning approach as a reference.

#### Caption Prompt

```text
You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [
    {"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}
  ],
  "borders": ["none"]
}

Rules:
- caption.long:
    - Describe the composition, key subjects, actions, setting, colors, lighting, and artistic style in detail. Include only information visible in the image.
    - Preserve visible text exactly, excluding watermarks. Do not translate, paraphrase, romanize, or normalize image text.
    - Describe text layout as much as possible: line breaks, ordering, relative position, alignment, size hierarchy, font/style, color, spacing, and how text blocks are arranged.
- caption.medium: compress long into one paragraph.
- caption.short: summarize the main subject, action, or relationship in one sentence.
- Do not describe watermarks in any caption field. Record them only in watermark.
- watermark:
    - Use normalized 0–1000 [y_min, x_min, y_max, x_max] coordinates relative to the full image.
    - Set label to the visible watermark text, or a short description if unreadable.
    - Use [] if there are no watermarks.
- borders:
    - Record evident borders or margins using one or more of "left", "right", "up", and "down".
    - Use ["none"] if there are no evident borders or margins. Do not combine "none" with other directions.
- All three caption fields must be non-empty strings.
- Treat the original caption as a reference. Correct any claims that are not supported by the image.
- Return only the JSON object, without Markdown or explanations.

Here is the original caption reference:
{{reference}}
```

#### Caption Model

We have tried several VLMs for captioning images for MOSS-Image-Base. We recommend Gemma 4, Qwen 3.5 27B, or Qwen 3.5 35B-A3B. We generally use the following request format:

```python
body = {
    "model": model_name,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    "n": 1,
    "stream": False,
    "max_tokens": 16384,
    "temperature": 1,
    "top_p": 0.95,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
}
```

We recommend enabling thinking mode for more accurate captions.

#### Tips

1. Adding relevant world knowledge can improve captions.
2. An agentic captioning pipeline can achieve high caption accuracy.
3. Better understanding helps improve generation.

### Few-Step Distillation

We are actively working on few-step distillation methods such as DMD. Stay tuned.
