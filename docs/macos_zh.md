# Apple Silicon / MPS 实验性支持

[English](macos.md)

通过 PyTorch Metal/MPS 在苹果芯片 Mac 上运行。本次有真实权重出图记录的配置为
**M1 Max / 32GB 统一内存 / SFT**。其他芯片、小内存配置、Base 和更早的依赖组合尚需实测。
没有做权重量化，32GB 也不是已证明的最低内存门槛。

## 安装与启动

使用原生 arm64 Python 3.12 和兼容的 macOS/PyTorch，建议为一个完整模型、Python 环境和
下载文件预留约 40GB 磁盘。从仓库根目录执行：

```bash
MOSS_PYTHON=python3.12 bash scripts/setup-mac.sh
.venv/bin/hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir models/MOSS-Image-SFT
bash scripts/gradio-mac.sh
```

打开 http://127.0.0.1:7860，先用 512×512、10 步、1 张图试跑。保持终端运行，加载和推理可能
需要数分钟。原图和 `parameters.json` 保存到 `outputs/gradio/`，按 Ctrl+C 停止服务。
再次使用只需启动脚本，不必重复安装下载。脚本只读本地模型并关闭 Gradio 分析上报。

可选安装脚本使用 `constraints-mac.txt` 中的本地实测版本：torch 2.14.1、diffusers 0.41.0、
transformers 5.20.1、accelerate 1.15.0、Gradio 6.30.0，不影响现有安装入口的依赖范围。

启动脚本默认 DiT FP16；比较 BF16 时：

```bash
MOSS_MPS_DIT_DTYPE=bfloat16 bash scripts/gradio-mac.sh
```

文本编码器使用 BF16，VAE 使用 FP32。直接调用 `mps_runtime.load_pipeline` 时，若导入前未设置
环境变量，默认 DiT BF16。MPS 页面自动采用分阶段加载，并禁用 CUDA offload 选择和 Compile DiT。

## 实现与待评审项

约 17GB 的 Qwen 编码器和约 10GB 的 DiT 轮流驻留 MPS。先处理正负提示词，释放编码器后再加载
DiT；最近八组提示词编码采用 LRU 缓存，仅改变 seed、尺寸或步数时可复用。

MPS 不支持时间步正弦编码中的 FP64，因此将这一小段计算放到 CPU，保留精度后传回 MPS。
独立 pipeline 位于下载的模型包中，当前用严格匹配的源码替换适配；源码变化会明确报错。
ComfyUI 模型包含相同的条件 CPU 计算修复，但本次未验证完整 ComfyUI 出图链路。

针对本地观察到的模型加载死锁，在清理缓存/同步前，通过回传 CPU 等待 Metal 复制完成。
当前方案会在 MPS 可用时包装 `torch.mps.empty_cache` 和 `torch.mps.synchronize`，影响进程内其他
调用者；这是实验方案中明确需要评审的部分。

## 实测与检查

| 样例 | 参数 | 完整生成调用耗时 |
| --- | --- | --- |
| 中文黑板 | 1024×1024、20 步、BF16、seed 42 | 320.362 秒 |
| 三张英文角色海报 | 768×1024、28 步、FP16 | 395.150 / 367.596 / 344.072 秒 |

[真实图片、提示词及原始参数](https://github.com/EdwinjJ1/moss-image-mac/tree/main/assets/samples)
仅作为验证证据，不代表同条件速度、精度对照或峰值内存测量。本贡献保留 CUDA 路径，但未进行
CUDA 硬件回归验证。

不下载模型的 CPU 回归检查：

```bash
python -m pip install -e '.[ui]' -c constraints-mac.txt
python -m unittest discover -s tests -p test_mps_runtime.py -v
```

覆盖模型切换、缓存淘汰、时间步运算、源码变更检查、设备路由及页面构建。实际 MPS 出图需另行
硬件验证。MPS 不可用时检查 arm64 Python 和系统兼容性；内存不足先降尺寸和张数；补丁不匹配时
请提供模型 revision 和报错，不要绕过检查。
