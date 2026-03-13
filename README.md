# ai_song

AI 唱歌 — 自动化歌曲翻唱 pipeline，从 URL 到成品一条命令搞定。

在 B站找一首翻唱 → 下载 → 分离人声/伴奏 → AI 声音转换 → 智能变调 → 三轨混音 → 质量评估 → 输出到 iCloud 手机直接听。

## 功能

- **一条命令端到端**：输入 URL + 模型路径，自动完成全部流程
- **Applio 引擎**：基于 RVC 的声音转换，兼容所有 RVC v1/v2 社区模型
- **伴唱转换**：主唱和伴唱都用同一音色转换，声音统一不割裂
- **智能变调**：F0 分析自动推荐最佳 transpose，让声音保持在模型甜区
- **自动质量评估**：UTMOSv2 自然度打分 + F0 pitch accuracy，减少人工试听量
- **三轨混音**：转换后主唱 + 转换后伴唱 + 原伴奏，带完整效果链（压缩/EQ/混响/Limiter）
- **macOS Apple Silicon 原生**：MPS 加速，M5 上 255s 歌曲转换 ~30s（8:1 实时比）

## 环境要求

- macOS + Apple Silicon（M1/M2/M3/M4/M5）
- Python 3.11+
- ~5GB 磁盘空间（模型权重 + 依赖）

## 安装

```bash
# 1. 克隆项目
git clone https://github.com/u03013112/ai_song.git
cd ai_song

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装基础依赖
pip install -e .

# 4. 安装 Applio（声音转换引擎）
git clone --depth 1 https://github.com/IAHispano/Applio.git third_party/Applio
pip install -r third_party/Applio/requirements.txt

# 5. 安装额外依赖
pip install audio-separator torchfcpe librosa soxr mir_eval
pip install git+https://github.com/sarulab-speech/UTMOSv2.git  # 质量评估（可选，781MB 模型）

# 6. 准备 RVC 模型
# 将 .pth 模型文件放到 models/ 目录下，例如：
# models/hayley_williams/model.pth
```

## 使用

### 完整 pipeline（推荐）

```bash
# 基础用法：URL + 模型
python -m ai_song "https://www.bilibili.com/video/BV1xxxxx" \
    --model models/hayley_williams/model.pth

# 推荐用法：启用智能变调 + 质量评估
python -m ai_song "https://www.bilibili.com/video/BV1xxxxx" \
    --model models/hayley_williams/model.pth \
    --auto-transpose \
    --evaluate

# 完整参数
python -m ai_song "https://www.bilibili.com/video/BV1xxxxx" \
    --model models/hayley_williams/model.pth \
    --index models/hayley_williams/model.index \
    --transpose 0 \
    --instrumental-shift 0 \
    --f0-method fcpe \
    --index-rate 0.0 \
    --auto-transpose \
    --evaluate \
    --output-dir output/ \
    --name "my_song_final.wav"
```

### 单步执行

```bash
# 下载
python -m ai_song.download "https://www.bilibili.com/video/BV1xxxxx" --output-dir output/downloads

# 分离人声/伴奏
python -m ai_song.separate --input song.wav --output-dir output/separated

# 声音转换
python -m ai_song.convert --input vocals.wav --model model.pth --transpose 0

# F0 分析 + 变调推荐
python -m ai_song.transpose --input vocals.wav --method fcpe

# 混音
python -m ai_song.mix vocals.wav instrumental.wav --output mixed.wav

# 质量评估
python -m ai_song.evaluate --input converted.wav --reference original.wav --transpose 1
```

### CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `url` | （必填） | B站/YouTube 视频 URL |
| `--model` | （必填） | RVC 模型路径（.pth） |
| `--index` | None | 特征索引路径（.index），无则设 index-rate=0 |
| `--transpose` | 0 | 人声变调（半音，±12） |
| `--instrumental-shift` | 0 | 伴奏变调（半音，±2 安全范围） |
| `--f0-method` | fcpe | F0 提取方法：fcpe / rmvpe / crepe |
| `--index-rate` | 0.0 | 特征检索比率（0-1，无 index 文件时必须为 0） |
| `--auto-transpose` | false | 自动分析 F0 并推荐最佳 transpose |
| `--evaluate` | false | 混音后运行质量评估（UTMOSv2 + pitch accuracy） |
| `--no-backing` | false | 跳过伴唱分离和转换 |
| `--no-icloud` | false | 不复制到 iCloud Drive |
| `--output-dir` | output/ | 输出目录 |
| `--name` | 自动生成 | 最终输出文件名 |

## Pipeline 流程

```
URL
 │
 ▼
[1] 下载 ─────────────── yt-dlp 提取音频为 WAV
 │
 ▼
[2] 分离 ─────────────── BS-RoFormer（SDR 12.96）→ 人声 + 伴奏
 │
 ▼
[2.5] Karaoke 分离 ──── MelBand-RoFormer → 主唱 + 伴唱
 │
 ▼
[2.8] F0 分析 ────────── FCPE 提取 F0 → 推荐 transpose（--auto-transpose）
 │
 ▼
[3] 声音转换 ─────────── Applio（RVC）主唱 + 伴唱分别转换
 │
 ▼
[4] 三轨混音 ─────────── 效果链处理 + LUFS 归一化 + 混音
 │
 ▼
[5] 质量评估 ─────────── UTMOSv2 + pitch accuracy（--evaluate）
 │
 ▼
输出 → iCloud Drive → iPhone 直接播放
```

## 效果链

### 单轨效果（Individual Track Processing）

主唱：Pre-Gain -3dB → HPF 80Hz → Compressor(-16dB/2.5:1) → EQ(body+presence+air) → Warmth → Limiter -3dB → Delay → Reverb(轻量)

伴唱：HPF 80Hz → Compressor(-20dB/2.0:1) → EQ → Limiter -6dB → Reverb(轻量)

### Bus Reverb（统一空间感）

三轨混合后施加共享 Bus Reverb（room=0.35, wet=0.10），让人声、伴唱、伴奏听起来在同一个声学空间内，避免"KTV贴伴奏"的割裂感。

设计理念：单轨混响只做最小限度的 artifact 平滑，空间感统一交给 Bus Reverb。

三轨 LUFS：主唱 -17 / 伴唱 -22 / 伴奏 -18

## 项目结构

```
ai_song/
├── __main__.py      # 完整 pipeline 入口
├── download.py      # 下载模块（yt-dlp）
├── separate.py      # 分离模块（audio-separator）
├── convert.py       # 声音转换（Applio/RVC）
├── transpose.py     # F0 分析 + 智能变调
├── evaluate.py      # 质量评估（UTMOSv2 + pitch accuracy）
├── mix.py           # 三轨混音 + 效果链
└── utils.py         # 工具函数
```

## 模型

项目使用 RVC 社区预训练模型（.pth），不含在仓库中。推荐来源：

- [QuickWick/Music-AI-Voices](https://huggingface.co/QuickWick/Music-AI-Voices)（877 个模型）
- [binant 系列仓库](https://huggingface.co/binant)

当前测试最佳：**Hayley Williams**（RVC v2，600 epochs，摇滚嗓，穿透力强）

## 版本历史

| 版本 | 主要变化 |
|------|---------|
| V1.0 | 基础 pipeline：下载→分离→RVC转换→混音 |
| V1.1 | FCPE 升级、效果链调优（J 版本）、三轨混音 |
| V1.2 | 21 模型 A/B 对比，Hayley Williams 选定 |
| V1.3 | Applio 引擎、伴唱转换、智能变调、自动质量评估、端到端验证 |
| V1.4 | Bus Reverb 统一空间感，解决人声与伴奏声学空间割裂问题 |

详细开发记录见 [TODO.md](TODO.md)。
