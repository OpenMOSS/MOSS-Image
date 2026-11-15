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
    <a href="README.md">简体中文</a> &nbsp;|&nbsp; <a href="README_EN.md">English</a> &nbsp;|&nbsp; <strong>日本語</strong> &nbsp;|&nbsp; <a href="../README.md">メイン README</a>
  </p>
</div>

# MOSS-Image on Google Cloud TPU (`torch_tpu`)

**[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** を **Google Cloud TPU (`TPU v6e-1`, 31.24 GiB)** 上のネイティブ PyTorch で動作させるリポジトリです。BF16 推論と FP32 VAE デコードに対応し、元の CUDA BF16 と同等の画質を実現しています。

> [!TIP]
> **メモリに関する推奨事項**：デバイスメモリが **40 GB 未満**（`TPU v6e-1` の `31.24 GiB` など）の環境で **BF16** モデルを使用する場合は、**`offload` の利用を推奨します**（デフォルト `--offload auto` で自動有効化）。テキスト側と拡散側の重みを同時に常駐させることはできません。

---

## 1. 画質比較（CUDA 原版 vs TPU 生成）

同一プロンプト・同一シード（`848×1200`、55 ステップ）での比較結果：

| 元の CUDA BF16 参照画像 | TPU `v6e-1` BF16 生成画像 |
| :---: | :---: |
| <img src="assets/cuda_bf16_ref.png" alt="元の CUDA BF16 参照画像" width="360"> | <img src="assets/tpu_v6e_pallas.png" alt="TPU v6e-1 BF16 生成画像" width="360"> |

- **画素一致度**：CUDA BF16 比でコサイン類似度 **0.9980**、平均画素誤差 **6.15 / 255**（`PSNR 26.80 dB`）；`drop_tpu` offload 使用時は `d2h` と完全一致（類似度 **1.0**、最大誤差 **0**）。

---

## 2. 生成速度・理論演算効率・Offload 性能

Google Colab **`TPU v6e-1`**（`31.24 GiB` HBM、`848×1200`、55 ステップ、CFG `3.0`）での実測値です。

### 各シナリオの画像生成・VAE 速度

| シナリオ | 合計時間 | 内訳 | ピーク HBM |
| :--- | :---: | :--- | :---: |
| **同一プロンプト連続生成** | **16.72s** | 55 ステップデノイズ 16.43s + VAE デコード等 ~0.29s | 16.33 GiB |
| **10 シード連続生成平均** *(2〜10 枚目)* | **16.73s** | 約 299 ms/step（初回コールド 95.56s 込みで 10 枚 246.17s） | 16.25 GiB |
| **プロンプト変更生成** *(drop_tpu)* | **37.71s** | モデル切替 20.75s + デノイズ 16.54s + エンコード/VAE ~0.42s | 20.67 GiB |
| **ディスクキャッシュヒット起動** | **41.62s** | ロード 2.20s + デノイズ 16.60s + 転送/VAE | 15.24 GiB |
| **初回コールド実行** *(最適化前基準)* | 88.42s | Pallas なし 115.17s；融合なし 124.19s | 15.79 GiB |

> 参考値：融合なし同一プロンプト実行は **49.08s**（デノイズ 48.61s）；コールド拡散 + VAE 単体は **35.56s**（融合なし 70.20s）；非同期 `d2h` 切替生成は **84.41s**。

### 理論演算効率（`TPU v6e-1` BF16 ピーク `918 TFLOPs` 対比）

- **画像拡散バックボーン（融合あり）**：**352.8 TFLOPs**（**38.4% MFU**）、未融合 eager（119.3 TFLOPs / 13.0% MFU）比で **2.96 倍高速**。
- **テキストエンコーダ（Pallas 最適化）**：単一層 **2.03 ms**（**3.08 倍高速**）、再帰カーネル **1.08 ms**（**4.57 倍高速**）；テキスト前向き推論はコールド **0.31s** / 長さ変更ウォーム **0.16s** / 同一反復 **0.09s**。

### `< 40 GB` 環境における Offload 切替速度

| Offload 方式 | プロンプト切替時間 | 生成合計時間 | 高速化率 |
| :--- | :---: | :---: | :---: |
| 同期転送 + 従来 `d2h` *(4 ステップ基準)* | 100.73s | 103.29s *(4 ステップ)* | 1.00x |
| 非同期 H2D + 従来 `d2h` *(55 ステップ)* | 67.08s | 84.41s | 1.50x |
| **非同期 H2D + `drop_tpu`** *(推奨・55 ステップ)* | **20.75s** | **37.71s** | **3.23x（同期比 4.86x）** |

> 同期転送 + `drop_tpu` は **48.96s**（2.06x）；Run 1 コールド切替も `49.00s` から **18.68s**（全体 35.72s）へ短縮。

---

## 3. クイックスタート

```bash
git clone https://github.com/OpenMOSS/MOSS-Image.git
cd MOSS-Image
bash tpu/setup-tpu.sh
pip install --no-deps deltanet-tpu

# 1. TPU 動作確認
python3 tpu/run_tpu.py --mode probe

# 2. 高速画像生成（演算子融合 + Pallas GDN；40 GiB 未満では自動的に offload が有効化）
python3 tpu/run_tpu.py \
  --mode generate --eager-mode defer_and_fuse --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --prompt "A ceramic teapot on linen" --output outputs/moss_tpu.png
```

> [!NOTE]
> `--gdn-pallas` には **Python ≥ 3.12** と、TorchTPU の固定版 `libtpu` に互換性のある **JAX** が必要です（Google Colab TPU などではプリインストール済み）。JAX 未導入または Python `< 3.12` の環境では、`--gdn-pallas` を外すことで通常の Eager テキストエンコードにフォールバックできます（生成自体は可能ですが、テキストエンコード速度が低下します）。

Google Colab では [`tpu/moss_image_torchtpu_colab.ipynb`](moss_image_torchtpu_colab.ipynb) から直接実行できます。詳細手順は [tpu/tpu.md](tpu.md) をご覧ください。

---

## 4. 技術詳細・データ（展開して表示）

<details>
<summary><strong>⚙️ 3 ステージ設計、drop_tpu、非同期 H2D、Pallas ブリッジ</strong></summary>

```text
Prompt テキスト
   │
   ├─► ディスクキャッシュ (--prompt-cache-dir) / CPU 8件 LRU ヒット ───────┐
   │                                                                       ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU 隠れ状態 [1, L, 4096]
                                                                           │
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, 静的バケット 512/1024/2048/4096)
     • 事前計算 FP32 timestep 正弦波 + 拡張済み [1, 1, Q, S] SDPA マスク
                                                                           │
                                                                           ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + NHWC→NCHW GroupNorm フック) ──► PNG
```

- **3 ステージ分割ランタイム（`tpu/tpu_runtime.py`）**：Stage 1 はパディングなしでエンコードし `[1, L_true, 4096]` を CPU に退避（8 件 LRU + `--prompt-cache-dir`）。`vae`（~0.3 GiB）は初回拡散ロード後、`diffusion <-> text` 切替中も TPU へ常駐します。
- **ゼロ D2H `drop_tpu`**：ホストメモリ `>= 36 GiB` で自動有効化。`_release_module_to_meta` でオフステージ重みを 0 バイト `meta` テンソルへ解放（`d2h = 0.0s`）し、ホスト RAM のスナップショットから直接復元します（`172.95 GiB` 環境で Run 2 後 RSS `30.91 GiB` / 空き `138.24 GiB`）。
- **非同期 `non_blocking=True` H2D**：単一の `synchronize()` にまとめて 400 テンソル（12.50 GiB）転送速度を `0.53 GiB/s`（23.78s）から **1.96 GiB/s**（6.38s、事前 pin で `2.05 GiB/s`）へ高速化しました。
- **約 `5.0 GiB` の残留 HBM 領域**：初回実行後 `d2h`（5.00–5.03 GiB）・`drop_tpu`（5.01–5.03 GiB）とも Python 側の生存 TPU テンソル `0` 個のまま約 `5.0 GiB` が残りますが、PJRT C++ ランタイムとコンパイルキャッシュの固定領域です。
- **Pallas GDN と Eager 自動切替（`tpu/tpu_qwen35.py`）**：24 層の `GatedDeltaNet` を `deltanet_tpu.chunk_gdn`（`128` 境界静的バケット）で実行（フォールバック 0 回：`96/96`・`48/48`）。`--text-eager-mode auto` が Stage 1 を `default` + Pallas、Stage 2/3 を `defer_and_fuse` に振り分けます。

</details>

<details>
<summary><strong>🛠️ ハードウェア不具合修正・Encode 詳細計測・単体テスト</strong></summary>

1. **4D SDPA マスク `NaN` 修正**：`[1, 1, 1, S]` を `[1, 1, Q, S]` へ拡張し `S >= 512` での全黒出力を解消。
2. **VAE `GroupNorm` 連続化フック**：`ensure_group_norm_contiguous(vae)` で NHWC 出力を NCHW へ連続化。
3. **`--compile` (`e0200`) 分離**：スケジューラ `.nonzero()` 同期で `vmem_address_out_of_range` が発生するためデフォルト `False`。

**Stage 1 `encode-bench`（329 正 + 24 負トークン、非同期 H2D 最適化前）**：

| モード | H2D | コールド | 負プロンプト | 長さ変更ウォーム | 同一反復 | ピーク HBM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `eager` (`DEFER_NEVER`) | 28.653s | 1.993s | 0.908s | 1.168s | 0.164s | 15.79 GiB |
| `defer_and_fuse` | 20.577s | 11.023s | 4.659s | 4.901s | 0.072s | 15.73 GiB |
| `eager+gdn_pallas` | 23.860s | **0.310s** | **0.930s** | **0.161s** | 0.092s | 15.83 GiB |
| `defer_and_fuse+gdn_pallas` | 26.311s | 6.672s | 1.684s | 1.859s | **0.049s** | 15.89 GiB |

- **数値一致度**：TPU HF vs TPU Pallas 画像 `cos = 0.999442`（平均差 `2.461/255`、`PSNR 32.31 dB`）；32 層埋め込み `cos = 0.999848`（正）/ `0.999758`（負）；TPU HF vs CUDA `cos = 0.997435`。
- **検証コマンド（32 件の単体テスト）**：
  ```bash
  ruff check tpu gradio_app.py inference_runtime.py tests/test_mps_runtime.py
  python3 -m unittest discover -s tpu/tests -p "test_tpu*.py" -v
  ```

</details>

<details>
<summary><strong>📁 ファイル一覧・今後のロードマップ（TPU 関連を `tpu/` に集約）</strong></summary>

| ファイル | 役割 |
| :--- | :--- |
| [`tpu/__init__.py`](__init__.py) | `tpu` パッケージ定義 |
| [`tpu/tpu_runtime.py`](tpu_runtime.py) | 3 ステージ実行、`drop_tpu` + 非同期 H2D + 常駐 VAE、静的バケット、SDPA/FP32/`GroupNorm` 修正 |
| [`tpu/tpu_qwen35.py`](tpu_qwen35.py) | Pallas `chunk_gdn` ブリッジ、`128` アライン因果パディング、永続キャッシュ |
| [`tpu/run_tpu.py`](run_tpu.py) | TPU CLI（`probe`, `encode-only`, `encode-bench`, `generate`, `benchmark`, `stage-switch-bench`） |
| [`tpu/bench_qwen35_gdn_tpu.py`](bench_qwen35_gdn_tpu.py) | 単体 GDN・単一層 A/B 計測スクリプト |
| [`tpu/setup-tpu.sh`](setup-tpu.sh) | TorchTPU 0.2.0 + PyTorch CPU 環境構築スクリプト |
| [`tpu/moss_image_torchtpu_colab.ipynb`](moss_image_torchtpu_colab.ipynb) | Google Colab ノートブック |
| [`tpu/tpu.md`](tpu.md) | 検証ログ・障害解析・ベンチマークレポート |
| [`tpu/tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tpu/tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 32 件の CPU 単体・一致テスト |
| [`gradio_app.py`](../gradio_app.py) / [`inference_runtime.py`](../inference_runtime.py) | CUDA / MPS / TorchTPU マルチバックエンド切替 |

**今後の最適化**：Pallas `SplashAttention`（38.4% MFU 超え）、`.nonzero()` 同期解消と `--compile` 有効化、CFG `B=2` 統合 + `128/256` バケット追加、`Qwen3.5-9B` INT8/W4 量子化による全モデル常駐。

</details>

---

## 5. 上流プロジェクト・引用

**[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** をベースにしています（[Apache-2.0 License](../LICENSE)）。CUDA/MPS/ComfyUI の利用方法は [USERGUIDE.md](../USERGUIDE.md) / [USERGUIDE_ZH.md](../USERGUIDE_ZH.md) をご覧ください。

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
