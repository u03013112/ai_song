# TODO

## V1：跑通完整 pipeline

### 阶段一：环境搭建

- [x] 初始化 Python 项目结构（pyproject.toml / requirements.txt）
- [x] 集成 `yt-dlp`，实现从 B站 URL 直接提取音频为 WAV 格式
- [x] 验证 macOS (Apple Silicon M5) 上 PyTorch MPS 后端可用

> **实测记录**：Python 3.11.3, PyTorch 2.10.0, MPS 正常。
> 测试下载 B站视频 BV1hpPszEEfv（《海阔天空》女烟嗓撕裂版），yt-dlp 提取 WAV 成功，5MB 压缩 → 43MB WAV（4分17秒）。

### 阶段二：声音分离

- [x] 集成 `audio-separator` 0.41.1，使用 BS-RoFormer 模型（SDR 12.0+，当前 SOTA）
- [x] 输出 vocals + instrumental 两轨（V1 不分主唱/伴唱，整体人声一起处理）
- [x] CLI 入口：`python -m ai_song.separate --input song.wav --output-dir output/`

> **实测记录**：模型 `model_bs_roformer_ep_368_sdr_12.9628`（610MB），首次需下载。
> 处理《海阔天空》4分17秒歌曲，分离用时 4分32秒（约 1:1 实时），效果良好。
> V1.5 切换 mlx-audio-separator 后预计可降至 ~3 秒。
> **已知问题**：人声部分有些段落处理不够干净，听感不够清晰。V1 先接受，后续版本尝试改进（可尝试更换模型、调整参数、或多模型 ensemble）。

### 阶段三：声音变换 & 变调

- [ ] 集成 RVC 推理，支持加载社区预训练模型（.pth）
- [ ] 支持 `transpose` 参数（半音为单位，范围 -12 ~ +12）
- [ ] 集成 `pedalboard`（Spotify 开源，内置 Rubberband），实现伴奏 pitch shift
- [ ] 支持混合变调策略：当差距较大时，伴奏和人声各分担一部分 pitch shift
  - 例：需要 +5 → 伴奏 shift +2，人声 transpose +3，两边都在安全范围内
  - 人声 transpose 安全范围：±4 半音以内质量几乎无损
  - 伴奏 pitch shift 安全范围：±2 半音以内几乎透明
- [ ] 实现批量模型试听：输入一段音频片段（如副歌），对多个模型依次推理，输出对比音频
- [ ] CLI 入口：`python -m ai_song.convert --input vocals.wav --model model.pth --transpose 0 --instrumental-shift 0`

### 阶段四：混音合成

- [ ] 响度对齐：将 vocals 和伴奏的 LUFS 匹配到统一标准
- [ ] 混响（reverb）：为转换后的人声添加混响，使其融入伴奏
- [ ] EQ 均衡（可选）：处理人声与伴奏的频段冲突
- [ ] CLI 入口：`python -m ai_song.mix --vocals converted.wav --instrumental instrumental.wav`

### 阶段五：质量评估 & 自动化

- [ ] 快速预览机制：支持只处理一段（如副歌 30s）快速判断效果，避免整首歌处理完才发现不行
- [ ] 完整 pipeline 串联：一条命令从 URL 到最终成品
  - `python -m ai_song --url <bilibili_url> --model model.pth --transpose -2 --instrumental-shift 2`
- [ ] 批量处理：支持多首歌、多个模型的组合批量生成
- [ ] 输出目录指向 iCloud Drive，iPhone 自动同步试听：
  - 输出路径：`~/Library/Mobile Documents/com~apple~CloudDocs/ai_song_output/`
  - iPhone 打开"文件" app → iCloud 云盘 → ai_song_output 直接播放

---

## V1.5：性能优化（可选）

- [ ] 将 `audio-separator` 切换为 `mlx-audio-separator`（MLX 原生，M5 上快 2.6 倍+）
- [ ] 验证模型兼容性（BS-RoFormer 等模型在 MLX 版本下效果一致）

---

## V2：伴唱分离 & 精细化 & 自动评估

- [ ] 引入 MelBand RoFormer Karaoke V2 模型，实现主唱/伴唱分离（同样通过 audio-separator 调用）
- [ ] 分离后主唱单独做 voice conversion，伴唱保留原声或另行处理
- [ ] 混音阶段适配三轨合并（转换后主唱 + 原伴唱 + 伴奏）
- [ ] "Bounce-Back" 变调技法：
  1. 用 pedalboard 将源人声 pitch shift 到模型甜区
  2. RVC 推理 transpose = 0（模型最高保真）
  3. 将转换后音频 shift 回原调
  - 让 RVC 永远在甜区工作，大跨度 shift 交给 DSP 工具处理
- [ ] 自动质量评估（用于批量模型试听场景，自动排序后人工只听 top N）：
  - F0 曲线对比（torchcrepe / RMVPE）：检测音准偏差，跑调是最致命问题
  - SingMOS-Pro：专为歌声训练的 MOS 预测，给"好听度"分数
  - UTMOSv2（可选）：对合成感/金属感/爆音等瑕疵敏感，快速筛掉明显有问题的

---

## 工具选型备忘

| 角色 | 工具 | 说明 |
|------|------|------|
| 模型工厂 | UVR (GUI) | 社区训练/发布模型的地方，不直接用于 pipeline |
| CLI 引擎 | audio-separator | UVR 模型的 Python/CLI 封装，V1 主力 |
| Apple Silicon 加速 | mlx-audio-separator | audio-separator 的 MLX 原生移植，V1.5 升级 |
| 经典基线 | demucs (Meta) | 四轨分离，被 RoFormer 超越，暂不使用 |
| 变调工具 | pedalboard (Spotify) | 内置 Rubberband，伴奏 pitch shift 用，比 librosa 快 300 倍 |
| 歌声评估 | SingMOS-Pro | 专为歌声训练的 MOS 预测，V2 自动评估主力 |
| 音准检测 | torchcrepe / RMVPE | F0 曲线提取与对比，检测跑调 |
| 通用音质 | UTMOSv2 | 对合成瑕疵敏感，辅助筛选 |

## 变调策略速查

| 差距 | 策略 | 做法 |
|------|------|------|
| ≤4 半音 | 纯 transpose | RVC transpose 直接搞定 |
| 5~8 半音 | 混合方案 | 伴奏 shift 一部分 + 人声 transpose 一部分，各不超安全范围 |
| >8 半音 | Bounce-Back (V2) | 源人声先 shift 到甜区 → RVC transpose=0 → 结果 shift 回原调 |
