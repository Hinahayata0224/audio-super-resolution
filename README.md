# Audio Super-Resolution / Restoration Pipeline

MP3/AAC 音频修复与超分辨率管道。将两个开源项目组合成一条两级处理链路：
先用 **Wavelet U-Net** 去除 MP3 压缩伪影（Stage 1），再用 **AudioSR** 扩散模型
做 48kHz 超分辨率（Stage 2）。

## 处理流程

```
输入音频 (低质量 MP3/AAC)
    │
    ▼  Stage 1: 小波去伪影 (44.1kHz)
    │   Wavelet U-Net 去除 MP3 压缩伪影
    │
    ▼  Stage 2: 超分辨率 (48kHz)
    │   AudioSR —— 基于扩散模型的音频超分
    │
    ▼
输出音频 (修复 + 超分)
```

## 目录结构

```
├── scripts/                      # 整合管道（自定义）
│   ├── restore.py                # 主入口：两级修复管道（Stage 1 + Stage 2）
│   ├── compare.py                # 原始/修复音频定量对比（LSD 对数谱距离）
│   └── verify_port.py            # 验证 PyTorch 移植版与原始 TF 版输出一致性
│
├── stages/                       # 按处理阶段划分的子项目
│   ├── stage1_wavelet_unet/      # 【开源项目①】matthewmcq/upscalemp3_v2 (MIT)
│   │   │                         # Wavelet U-Net MP3 增强，TensorFlow/Keras 实现
│   │   ├── src/                  # 模型、训练、推理（TF）
│   │   ├── README.md             # 原项目文档
│   │   └── LICENSE               # MIT License
│   │
│   └── stage1_wavelet_unet_pytorch/   # 【移植版】upscalemp3_v2 的 PyTorch 重实现
│       │                         # 零 TF 依赖，权重从 Keras h5 加载
│       └── src/                  # config / dwt / blocks / model / inference / weights
│
├── models/                       # 模型权重缓存（未入库，见"模型下载"）
└── requirements.txt              # 根目录依赖（Stage 2 AudioSR + 音频处理）
```

> `stage1_wavelet_unet/` 是 Stage 1 的 TF/Keras 原始实现，`stage1_wavelet_unet_pytorch/`
> 是其 PyTorch 移植版（推理管道实际使用移植版）。`stages/` 目录只含代码，不含权重。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型权重（见下方"模型下载"）

# 3. 运行修复管道
python scripts/restore.py -i "music.mp3"                        # 完整两级管道
python scripts/restore.py -i "music.mp3" --stage1 skip          # 仅 Stage 2 超分
python scripts/restore.py -i "music.mp3" -o ./out --ddim_steps 100
```

## 性能优化说明

- **Stage 1 批量推理**：`stage1_wavelet_unet_pytorch` 按 1s 段切分后以
  `batch_size=8` 批量前向（`inference.py`），显著提升 Conv1D 吞吐，
  数学结果与逐段串行完全一致。
- **Stage 2 chunk 并行**：`restore.py` 用线程池并行处理 10.24s 的音频块
  （AudioSR 的 torch 推理会释放 GIL，多核 CPU 上可并行，默认 4-8 workers）。
  并行不改变 overlap-add 结果。
- **逐 chunk 增益对齐**：Stage 2 每个 chunk 输出按输入 RMS 归一化，
  避免 AudioSR 输出归一化导致的 chunk 边界响度跳变。

## 模型下载

模型权重体积较大，未包含在仓库中（见 .gitignore），需手动下载：

| 模型 | 用途 | 来源 | 大小 |
| --- | --- | --- | --- |
| AudioSR (`pytorch_model.bin`) | Stage 2 超分（basic/music） | [HaoheLiu/audiosr](https://huggingface.co/spaces/haoheliu/audiosr) | ~5.8 GB |
| roberta-base | AudioSR 文本条件编码器 | [roberta-base](https://huggingface.co/roberta-base) | ~500 MB |
| `model_13M.weights.h5` | Stage 1 小波去伪影（PyTorch 移植版） | 由原项目训练/发布 | 147 MB |

放置路径（与代码内默认路径一致）：

```
models/audiosr/pytorch_model.bin                    # Stage 2
models/huggingface/roberta-base/...                 # Stage 2 文本编码器缓存
stages/stage1_wavelet_unet_pytorch/weights/model_13M.weights.h5   # Stage 1
```

> AudioSR 权重也可通过 HuggingFace `huggingface-cli download` 或直接运行
> `scripts/restore.py`（去掉 `HF_HUB_OFFLINE=1` 时自动下载）获取。

## 子项目与许可证

本仓库由以下开源项目组合而成，各自保留原始许可证：

| 组件 | 原项目 | 许可证 |
| --- | --- | --- |
| `stages/stage1_wavelet_unet/` | [matthewmcq/upscalemp3_v2](https://github.com/matthewmcq/upscalemp3_v2) — Wavelet U-Net MP3 增强 | MIT |
| `stages/stage1_wavelet_unet_pytorch/` | upscalemp3_v2 的 PyTorch 移植（自研封装） | 同 MIT |
| Stage 2 超分 | [haoheliu/audiosr](https://github.com/haoheliu/audiosr) — Audio Diffusion Super-Resolution | 见原项目 |

## 备注

- `stages/stage1_wavelet_unet/` 为原项目文件（含其 LICENSE），未做修改；其内嵌 git 历史已在整理时移除。
- 使用 `hf-mirror.com` 镜像可加速 HuggingFace 权重下载（`scripts/restore.py` 默认已设置）。
