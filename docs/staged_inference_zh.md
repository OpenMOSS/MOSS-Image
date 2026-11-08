# 分开运行提示词编码与图像生成

[English](staged_inference.md)

`moss_batch` 将提示词编码保存到磁盘，再由另一个 Python 进程生成图像。编码进程加载文本编码器和 tokenizer；生成进程只加载 transformer、scheduler 和 VAE，不加载文本编码器或 tokenizer。你可以复用编码来调整 seed 和采样参数，也可以把编码目录复制到另一台机器生成。

现有管线已经支持 CPU offload。这套工作流减少重复文本编码，并让生成阶段无需加载文本编码器；实际内存占用和速度取决于硬件、参数及 offload 模式。它不修改模型或采样算法。

## 运行两个阶段

在仓库根目录、已准备好的推理环境中执行：

```sh
python -m pip install -e .
python -m moss_batch validate --jobs examples/staged_jobs.jsonl
```

`validate` 只检查任务清单，不导入推理依赖，也不下载模型。示例包含两个提示词相同、seed 和 CFG 不同的任务；编码只计算一次，并分别保存到两个任务 ID 对应的目录。

从 Hugging Face 模型历史中选择一个 commit，替换下面的 `MODEL_COMMIT_SHA`。两个命令使用相同的模型和 commit：

```sh
python -m moss_batch encode --jobs examples/staged_jobs.jsonl --model OpenMOSS-Team/MOSS-Image-Base --revision MODEL_COMMIT_SHA --cache-dir encodings
python -m moss_batch generate --jobs examples/staged_jobs.jsonl --model OpenMOSS-Team/MOSS-Image-Base --revision MODEL_COMMIT_SHA --cache-dir encodings --output-dir outputs
```

等待 `encode` 退出后再运行 `generate`。两条命令分别启动进程，因此编码阶段占用的模型内存在生成开始前已释放。使用 Hub 模型时，每个阶段只下载所需组件目录和管线代码。不传 `--revision` 时，命令会解析并打印当前 Hub revision；固定 commit 可以避免两个阶段之间的上游更新使缓存失效。

默认使用 CUDA 和 `--offload model`。`--offload sequential` 使用逐层 CPU offload；`--offload none` 将该阶段的组件放在指定设备上。CPU 执行需要同时指定 `--device cpu --offload none`。加上 `--local-files-only` 可只使用已有的 Hub 本地缓存，不下载缺少的组件。

## 任务清单

清单为 UTF-8 JSONL，每行一个扁平 JSON 对象，每个任务输出一张图片。允许空行。`id` 和 `prompt` 必填；未知字段和重复 ID 会被拒绝。

| 字段 | 默认值 | 要求 |
| --- | --- | --- |
| `id` | 必填 | 1–128 个 ASCII 字母、数字、`_` 或 `-`，以字母或数字开头；忽略大小写后唯一；不能使用 Windows 保留设备名 |
| `prompt` | 必填 | 非空字符串 |
| `negative_prompt` | `""` | 字符串 |
| `seed` | `-1` | `-1` 表示选择并保存随机 seed；否则为 `0` 至 `2**53 - 1` 的整数 |
| `height`、`width` | `1024` | 16 的正整数倍 |
| `num_inference_steps` | `50` | 正整数 |
| `guidance_scale` | `3.0` | 有限、非负数 |
| `shift` | `5.0` | 有限正数 |
| `max_sequence_length` | `4096` | 1 至 4096 的整数 |
| `uncond_skip_layers` | `[22]` | 不重复的非负层索引数组；`[]` 关闭这项跳层 |
| `slg_layers` | `[]` | 不重复的非负层索引数组 |
| `slg_scale` | `0.0` | 有限、非负数 |
| `slg_start`、`slg_stop` | `0.0`、`1.0` | `0 <= slg_start <= slg_stop <= 1` |

生成前会根据已加载 transformer 的深度检查层索引。提示词中的 Unicode、空白，以及作为字符串传入的 Layout JSON 都会原样保留，不添加前缀或自动扩写。即使 `guidance_scale` 为 1，也会编码正、负两个分支，方便后续调整 CFG。

字段沿用管线的名称（`num_inference_steps`、`guidance_scale`），与 Gradio 的 `steps`、`cfg` 不同。Gradio 参数文件不能直接用作批量任务清单。

## 复用编码与跨机器使用

每个 `encodings/<id>/` 包含 `encoding.safetensors` 和 `metadata.json`。元数据记录模型来源、完整提示词、最大序列长度、dtype、编码器与 tokenizer 类、chat template 和编码时的依赖版本。加载时会检查张量形状、dtype、attention mask 和数值是否有限。

要尝试其他 seed、CFG、步数、shift 或图像尺寸，保留任务 ID 和编码相关字段，修改采样参数，然后用**新的 `--output-dir`** 运行 `generate`，即可复用编码。修改正向提示词、负向提示词、最大序列长度或模型来源后，需要重新编码：使用新的缓存目录或任务 ID，再运行 `encode`。已有且不兼容的缓存不会被覆盖。

要换机器生成，复制完整编码目录和任务清单，在目标机器安装本包，并指定相同的 Hub 模型与固定 commit。生成阶段下载 transformer、scheduler、VAE 和管线代码，无需下载文本编码器权重。

使用本地模型目录时，以可跨机器使用的版本标签代替 `--revision`：

```sh
python -m moss_batch encode --jobs examples/staged_jobs.jsonl --model /path/to/model --local-model-version my-bundle-v1 --cache-dir encodings
python -m moss_batch generate --jobs examples/staged_jobs.jsonl --model /path/to/model --local-model-version my-bundle-v1 --cache-dir encodings --output-dir outputs
```

生成端的模型目录需要官方管线代码、model index、transformer、scheduler 和 VAE。两台机器的路径可以不同，但相同版本标签应对应相同来源的模型文件。模型、tokenizer 或管线文件有任何更新，都需要更换标签。标签由使用者指定，工具不会验证同标签的两个目录是否包含完全相同的文件。本地目录和 Hub 使用不同的来源身份，即使文件来自同一模型，也不能直接混用这两种身份的缓存。

## 中断后继续

再次运行相同的 `generate` 命令即可。每个任务开始推理前，`outputs/<id>/parameters.json` 会保存实际 seed 和参数。图片先以原子操作写入 `01.png`，随后任务才标记为完成。只有参数和来源匹配、PNG 能完整解码且尺寸正确时，已完成任务才会被跳过。图片缺失或损坏会使用已保存的 seed 重新生成。参数或来源发生变化时会拒绝覆盖；新一轮尝试请使用新的输出目录。

输出记录使用本工具的 schema version 1，包含任务、编码和模型来源、实际 seed、完成状态及图片文件名。它与 Gradio 参数格式独立；元数据保存在 JSON 文件中，不嵌入 PNG。保存 seed 便于重放，但不同硬件或软件版本之间不保证逐像素一致。

每条命令通过 `.moss-batch.lock` 锁定自己的目标根目录。正常退出、已处理的异常和 Ctrl+C 都会释放锁。进程被强制终止后，先确认没有写入进程正在运行，再手动删除残留锁。缓存损坏或不兼容时，可换用新的缓存路径，或确认后只删除受影响任务的缓存目录，再运行 `encode`。工具不会自动删除或修复已有缓存。
