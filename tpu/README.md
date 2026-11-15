<div align="center">
  <p>
    <img src="../assets/lockup-light.svg#gh-light-mode-only" alt="MOSS-Image" width="460">
    <img src="../assets/lockup-dark.svg#gh-dark-mode-only" alt="MOSS-Image" width="460">
  </p>
  <p>
    <a href="moss_image_torchtpu_colab.ipynb"><img alt="Colab TPU" src="https://img.shields.io/badge/Colab-TPU%20v6e--1-F9AB00?style=flat-square&amp;logo=googlecolab&amp;logoColor=white"></a>
    &nbsp;
    <a href="tpu.md"><img alt="TorchTPU Docs" src="https://img.shields.io/badge/Docs-TorchTPU-4285F4?style=flat-square&amp;logo=googlecloud&amp;logoColor=white"></a>
    &nbsp;
    <a href="../README.md"><img alt="MOSS-Image Home" src="https://img.shields.io/badge/MOSS--Image-Main%20README-0f766e?style=flat-square"></a>
  </p>
  <p>
    <strong>简体中文</strong> &nbsp;|&nbsp; <a href="README_EN.md">English</a> &nbsp;|&nbsp; <a href="README_JA.md">日本語</a> &nbsp;|&nbsp; <a href="../README.md">主仓库主页</a>
  </p>
</div>

# MOSS-Image on Google Cloud TPU (`torch_tpu`)

在 **Google Cloud TPU (`TPU v6e-1`, 31.24 GiB)** 上原生运行 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)，支持完整 BF16 推理与 FP32 图像解码，画质与原版 CUDA 保持一致。

> [!TIP]
> **显存建议**：设备显存小于 **40 GB** 时，使用 **BF16** 精度模型建议开启 **`offload`**（默认 `--offload auto` 会自动开启），否则无法将文本与生图权重同时放进单卡显存。

---

## 一、画质对比（原版 CUDA vs TPU）

同提示词、同随机种子（`848×1200`、55 步）直接对比：

| 原版 CUDA BF16 参考图 | TPU `v6e-1` BF16 生成图 |
| :---: | :---: |
| <img src="assets/cuda_bf16_ref.png" alt="原版 CUDA BF16 参考图" width="360"> | <img src="assets/tpu_v6e_pallas.png" alt="TPU v6e-1 BF16 生成图" width="360"> |

- **像素一致性**：与原版 CUDA BF16 余弦相似度 **0.9980**，平均像素差 **6.15 / 255**（`PSNR 26.80 dB`）；开启 `drop_tpu` offload 后输出完全无损（余弦 **1.0**，最大像素差 **0**）。

---

## 二、生图速度、算子效率与 Offload 表现

以下全部实测于 Google Colab **`TPU v6e-1`**（`31.24 GiB` 显存，`848×1200` 分辨率，55 步，CFG `3.0`）。

### 1. 各场景图像生成与 VAE 速度

| 使用场景 | 总耗时 | 耗时分布 | 峰值显存 |
| :--- | :---: | :--- | :---: |
| **同词热态续生** | **16.72s** | 去噪 16.43s + VAE 解码与后处理 ~0.29s | 16.33 GiB |
| **10 连跑均值** *(第 2–10 张)* | **16.73s** | 单步 ~299 ms（10 张共 246.17s，含首张冷跑 95.56s） | 16.25 GiB |
| **换提示词生图** *(drop_tpu Offload)* | **37.71s** | 模型切换 20.75s + 去噪 16.54s + 编码/VAE ~0.42s | 20.67 GiB |
| **冷启动读取磁盘缓存** | **41.62s** | 权重加载 2.20s + 去噪 16.60s + 搬运/VAE | 15.24 GiB |
| **首次冷跑** *(优化前基线)* | 88.42s | 无 Pallas 加速 115.17s；无算子融合 124.19s | 15.79 GiB |

> 其他参考口径：关闭算子融合时同词热生为 **49.08s**（去噪 48.61s）；冷态仅扩散 + VAE 为 **35.56s**（无融合 70.20s）；开启异步传输的传统 `d2h` 换词生图为 **84.41s**。

### 2. 理论算子效率（对比 TPU 峰值 `918 TFLOPs` BF16）

- **图像扩散主干（开启算子融合）**：实测 **352.8 TFLOPs**，理论算力利用率 **38.4% MFU**（比未融合的 119.3 TFLOPs / 13.0% MFU 快 **2.96 倍**）。
- **文本编码算子（Pallas 加速）**：单层 **2.03 ms**（快 **3.08 倍**），递推核心 **1.08 ms**（快 **4.57 倍**）；文本前向编码冷态 **0.31s**、换词热态 **0.16s**、重复输入 **0.09s**。

### 3. `< 40 GB` 显存下的 Offload 切换速度

`Qwen3.5-9B`（15.63 GiB）与扩散 + VAE 模型（9.70 GiB）合计超过 31.24 GiB，因此更换提示词时会触发一次 offload 切换：

| Offload 方案 | 换词模型切换耗时 | 换词生图总耗时 | 提速对比 |
| :--- | :---: | :---: | :---: |
| 同步传输 + 传统 `d2h` *(4 步基线)* | 100.73s | 103.29s *(4 步)* | 1.00x |
| 异步传输 + 传统 `d2h` *(55 步)* | 67.08s | 84.41s | 1.50x |
| **异步传输 + `drop_tpu`** *(推荐，55 步)* | **20.75s** | **37.71s** | **快 3.23x（较同步快 4.86x）** |

> 仅开 `drop_tpu` 未开异步传输为 **48.96s**（2.06x）；Run 1 首次切换耗时也由 49.00s 降至 **18.68s**（整轮 35.72s）。

---

## 三、快速上手

```bash
git clone https://github.com/OpenMOSS/MOSS-Image.git
cd MOSS-Image
bash tpu/setup-tpu.sh
pip install --no-deps deltanet-tpu

# 1. 硬件连通性检测
python3 tpu/run_tpu.py --mode probe

# 2. 快速生图（< 40 GiB 显存自动启用 offload）
python3 tpu/run_tpu.py \
  --mode generate --eager-mode defer_and_fuse --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --prompt "A ceramic teapot on linen" --output outputs/moss_tpu.png
```

也可直接在 Colab 打开 [`tpu/moss_image_torchtpu_colab.ipynb`](moss_image_torchtpu_colab.ipynb)。完整参数与压测命令见 [tpu/tpu.md](tpu.md)。

---

## 四、技术原理与详细数据（按需展开）

<details>
<summary><strong>⚙️ 三阶段调度、drop_tpu 卸载、异步 H2D 与 Pallas 加速原理</strong></summary>

```text
Prompt 文本
   │
   ├─► 命中磁盘缓存 (--prompt-cache-dir) / 内存 8 条 LRU ─────────────────┐
   │                                                                      ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU 隐状态 [1, L, 4096]
                                                                          │
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, 静态桶 512/1024/2048/4096)
     • 预计算 FP32 timestep 正余弦编码 + 显式扩展 [1, 1, Q, S] SDPA 掩码
                                                                          │
                                                                          ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + NHWC→NCHW GroupNorm 钩子) ──► PNG
```

- **三阶段分时调度（`tpu/tpu_runtime.py`）**：Stage 1 按真实长度执行无填充编码（避免破坏因果状态），将 `[1, L_true, 4096]` 隐状态卸载至 CPU（8 条 LRU `_prompt_cache` + 可选 `--prompt-cache-dir`）；Stage 2/3 执行扩散与解码。`vae`（~0.3 GiB）在首次进入扩散后常驻 TPU，不在换词时重复卸载。
- **`drop_tpu` 零回拷卸载**：当主机内存 `>= 36 GiB` 时自动启用。CPU 内存保留只读权重快照，离场模块通过 `_release_module_to_meta` 直接释放在 TPU 上的参数为 0 字节 `meta` 张量（`d2h = 0.0s`），进场时从 CPU 快照直接恢复。在 172.95 GiB 主机内存上，Run 2 结束进程 RSS 为 30.91 GiB，仍余 138.24 GiB 可用内存。
- **异步 `non_blocking=True` H2D**：将参数与 buffer 拷贝合并至单次 `torch.tpu.synchronize()`，400 张量（12.50 GiB）合成传输速度由 0.53 GiB/s（23.78s）提升至 **1.96 GiB/s**（6.38s；预固定内存 2.05 GiB/s / 6.10s）。
- **`~5.0 GiB` 残留显存平台**：首次完整推理后 `torch.tpu.memory_allocated()` 会稳定保留约 `5.0 GiB`（`d2h` 5.00–5.03 GiB，`drop_tpu` 5.01–5.03 GiB），此时 Python `gc.get_objects()` 中存活 TPU 张量为 `0`，属于 TorchTPU/PJRT C++ 运行时与编译缓存的一次性常驻占用。
- **Pallas GDN 与分阶段 Eager 策略（`tpu/tpu_qwen35.py`）**：24 层 `GatedDeltaNet`（`Hk=16, Hv=32, D=128`）通过 `torch.tpu.pallas.jax_op` 接入 `deltanet_tpu.chunk_gdn` 与 `128` 对齐静态桶，实测 0 回退（`96/96` / `48/48`）。`--text-eager-mode auto` 会自动让 Stage 1 走 `default`（`DEFER_NEVER`）+ Pallas、Stage 2/3 走 `defer_and_fuse`。

</details>

<details>
<summary><strong>🛠️ TPU v6e-1 底层问题修复、Stage 1 微基准与单元测试</strong></summary>

1. **修复 4D 广播 SDPA 掩码 `NaN` 全黑图**：将 `text_attn_mask` 与 `joint_attn_mask` 从 `[1, 1, 1, S]` 扩展至 `[1, 1, Q, S]`，规避 `torch_tpu` Mosaic/cuDNN 在 `S=30` 误差及 `S >= 512` 产出 `NaN` 的问题。
2. **修复 VAE `GroupNorm` 崩溃**：`ensure_group_norm_contiguous(vae)` 在 `F.group_norm` 调用 `.view(...)` 前将 NHWC 卷积输出转为连续 NCHW。
3. **隔离 `--compile` `e0200` 核崩溃**：`transformer.compile(backend="tpu", fullgraph=True)` 会在调度器 `.nonzero()` D2H 同步处触发 `vmem_address_out_of_range`，默认关闭（`--compile=False`）。

**Stage 1 `encode-bench` 微基准（329 正向 + 24 负向 tokens，异步 H2D 前采集）**：

| 模式 | H2D 搬运 | 正向冷编码 | 负向编码 | 换长度热编码 | 同长度热编码 | 峰值 HBM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `eager` (`DEFER_NEVER`) | 28.653s | 1.993s | 0.908s | 1.168s | 0.164s | 15.79 GiB |
| `defer_and_fuse` | 20.577s | 11.023s | 4.659s | 4.901s | 0.072s | 15.73 GiB |
| `eager+gdn_pallas` | 23.860s | **0.310s** | **0.930s** | **0.161s** | 0.092s | 15.83 GiB |
| `defer_and_fuse+gdn_pallas` | 26.311s | 6.672s | 1.684s | 1.859s | **0.049s** | 15.89 GiB |

- **数值对齐**：TPU HF vs TPU Pallas 整图余弦 `0.999442`（平均差 `2.461/255`，`PSNR 32.31 dB`）；32 层隐状态正向 `cos = 0.999848`、负向 `cos = 0.999758`；未开 Pallas 相对 CUDA BF16 余弦 `0.997435`。
- **代码检查与 32 项单元测试**：
  ```bash
  ruff check tpu gradio_app.py inference_runtime.py tests/test_mps_runtime.py
  python3 -m unittest discover -s tpu/tests -p "test_tpu*.py" -v
  ```

</details>

<details>
<summary><strong>📁 文件清单与后续优化方向（TPU 代码统一收口于 `tpu/`）</strong></summary>

| 文件 | 作用 |
| :--- | :--- |
| [`tpu/__init__.py`](__init__.py) | `tpu` 包入口 |
| [`tpu/tpu_runtime.py`](tpu_runtime.py) | 三阶段流水线、`drop_tpu` + 异步 H2D + 常驻 VAE、静态桶、SDPA/FP32/`GroupNorm` 补丁 |
| [`tpu/tpu_qwen35.py`](tpu_qwen35.py) | Pallas `chunk_gdn` 桥接、`128` 对齐因果填充与磁盘提示词缓存 |
| [`tpu/run_tpu.py`](run_tpu.py) | TPU CLI（`probe` / `encode-only` / `encode-bench` / `generate` / `benchmark` / `stage-switch-bench`） |
| [`tpu/bench_qwen35_gdn_tpu.py`](bench_qwen35_gdn_tpu.py) | 独立 GDN 递推核与单层性能/精度 A/B 脚本 |
| [`tpu/setup-tpu.sh`](setup-tpu.sh) | TorchTPU 0.2.0 + PyTorch CPU 安装脚本 |
| [`tpu/moss_image_torchtpu_colab.ipynb`](moss_image_torchtpu_colab.ipynb) | Google Colab 一键运行 Notebook |
| [`tpu/tpu.md`](tpu.md) | 完整调试记录、报错分析与压测日志 |
| [`tpu/tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tpu/tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 32 项本地 CPU 单元测试与数值校验 |
| [`gradio_app.py`](../gradio_app.py) / [`inference_runtime.py`](../inference_runtime.py) | CUDA / MPS / TorchTPU 多后端路由入口 |

**后续路线图**：
1. 用 Pallas `SplashAttention` 替代 4D 掩码 `SDPA`，继续拉高 38.4% MFU；
2. 移除调度器 `.nonzero()` 同步并排查 `e0200` 以解锁 `--compile`；
3. 合并 CFG `batch=2` 并下探 `128, 256` 提示词桶；
4. 探索 `Qwen3.5-9B` INT8/W4 量化，使全部权重同时驻留 `31.24 GiB` 彻底消灭切换搬运。

</details>

---

## 五、上游项目与引用

基于 **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** 构建（[Apache-2.0 License](../LICENSE)）。原版 CUDA/MPS/ComfyUI 指南见 [USERGUIDE_ZH.md](../USERGUIDE_ZH.md) 与 [USERGUIDE.md](../USERGUIDE.md)。

```bibtex
@misc{ai2026mossimagetechnicalreport,
      title={MOSS-Image Technical Report},
      author={MOSI. AI},
      year={2026},
      eprint={2610.23374},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.23374},
}
```
