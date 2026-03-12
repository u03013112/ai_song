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

#### 伴唱分离（Karaoke Separation）

- [x] 使用 `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` 模型（913MB），对已分离的人声做二次分离
- [x] 输出 lead vocals（主唱）+ backing vocals（伴唱）两轨
- [x] 用户试听确认：**分离效果良好** ✅

> **实测记录**：模型 `mel_band_roformer_karaoke`（913MB），首次需下载（约 5 分钟）。
> 处理《海阔天空》5分26秒人声，分离用时 2分56秒（41 chunks × ~4.27s/chunk），MPS 加速。
> 注意文件名含 Unicode 特殊字符时，需用 pathlib glob 匹配而非直接拼接字符串。
> **耗时对比**：比第一步人声/伴奏分离（BS-RoFormer, 4分32秒处理4分17秒音频）稍慢，因为 karaoke 模型更大（913MB vs 610MB）且在纯人声中区分主唱/伴唱难度更高。
> **自动化建议**：对于无伴唱的歌曲，分离后伴唱文件能量极低（几乎静音），可通过 RMS/dBFS 阈值检测自动跳过伴唱处理流程。

### 阶段三：声音变换 & 变调

- [x] 集成 RVC 推理，支持加载社区预训练模型（.pth）
- [x] 支持 `transpose` 参数（半音为单位，范围 -12 ~ +12）
- [x] 集成 `pedalboard`（Spotify 开源，内置 Rubberband），实现伴奏 pitch shift
- [x] 支持混合变调策略：当差距较大时，伴奏和人声各分担一部分 pitch shift
  - 例：需要 +5 → 伴奏 shift +2，人声 transpose +3，两边都在安全范围内
  - 人声 transpose 安全范围：±4 半音以内质量几乎无损
  - 伴奏 pitch shift 安全范围：±2 半音以内几乎透明
- [ ] 实现批量模型试听：输入一段音频片段（如副歌），对多个模型依次推理，输出对比音频
- [x] CLI 入口：`python -m ai_song.convert --input vocals.wav --model model.pth --transpose 0 --instrumental-shift 0`

> **实测记录**：使用 Ariana Grande RVC v1 模型（52MB），`index_rate=0` 绕过 faiss-cpu Apple Silicon SEGFAULT。
> 4分17秒完整人声转换用时 31.8 秒（约 8:1 速度比），MPS 加速正常。
> **已知限制**：`faiss-cpu` 在 ARM64 macOS 上 SEGFAULT，`index_rate` 暂设为 0（不使用 .index 特征检索），timbre 还原度略低但可接受。
> **已知问题**（Ariana Grande + 《海阔天空》女烟嗓）：
> 1. 高音部分效果差，出现断音/失真——超出模型训练音域时 RVC 推理质量急剧下降
> 2. 伴唱表现差，很多时候不能清楚发音——可能与 `index_rate=0`（无特征检索）及模型本身对中文适配不佳有关
> 3. **疑似输入源问题**：原版为沙哑撕裂唱法，沙哑声带振动不规则导致 F0 提取跳变断裂，Hubert 特征谐波弱噪声多编码质量下降，高音+沙哑双重叠加放大失真。RVC 最适合清亮稳定的人声输入——需换清亮唱法版本对比验证

### 阶段四：混音合成

- [x] 响度对齐：将 vocals 和伴奏的 LUFS 匹配到统一标准（pyloudnorm）
- [x] 混响（reverb）：为转换后的人声添加混响，使其融入伴奏（pedalboard Reverb）
- [x] 高通滤波：去除人声低频杂音（pedalboard HighpassFilter, 80Hz）
- [x] 采样率自动重采样：RVC 输出 40kHz → 44.1kHz（scipy.signal.resample）
- [x] 防削波处理：检测并自动归一化 peak > 1.0
- [x] CLI 入口：`python -m ai_song.mix vocals.wav instrumental.wav`

> **实测记录**：混音处理 4分17秒歌曲用时 1.8 秒，输出 -13.2 LUFS。
> 人声 -16 LUFS + 伴奏 -18 LUFS，人声略突出，符合流行音乐混音习惯。

### 阶段五：质量评估 & 自动化

- [ ] 快速预览机制：支持只处理一段（如副歌 30s）快速判断效果，避免整首歌处理完才发现不行
- [x] 完整 pipeline 串联：一条命令从 URL 到最终成品
  - `python -m ai_song <bilibili_url> --model model.pth --transpose -2 --instrumental-shift 2`
- [ ] 批量处理：支持多首歌、多个模型的组合批量生成
- [x] 输出目录指向 iCloud Drive，iPhone 自动同步试听：
  - 输出路径：`~/Library/Mobile Documents/com~apple~CloudDocs/ai_song_output/`
  - iPhone 打开"文件" app → iCloud 云盘 → ai_song_output 直接播放

---

## V1.1：工具升级 & 效果调优 ✅ 已完成

> 目标：换掉过时/有问题的工具，用新歌源测试，调出满意的效果。
> 语音转换引擎暂不整体替换（GPT-SoVITS / DDSP-SVC 留到 V2），先在 RVC 框架内最大化效果。
> **状态：已完成** — 三轨混音（转换后主唱 + 原伴唱 + 伴奏）产出最终成品。

### 阶段一：工具升级（先升级再干活）

- [ ] **升级 faiss-cpu → v1.14.1**：修复 ARM64 SEGFAULT，启用 `index_rate=0.7`
  - 尝试 `pip install faiss-cpu==1.14.1`
  - 若仍 SEGFAULT → `conda install -c conda-forge faiss-cpu=1.14.1`
  - 若仍失败 → 用 hnswlib 替代（改 `vc_infer_pipeline.py` 的 `index.search()`）
  - 验证标准：用 Ariana Grande 模型 + test_5s.wav 跑 `index_rate=0.7` 不崩溃
- [x] **升级 F0 提取为 FCPE**：`pip install torchfcpe`，monkey-patch Pipeline.get_f0 支持 FCPE
  - FCPE 上下文感知能减少转音/呼吸处的断音
  - 验证标准：同一段音频分别用 RMVPE 和 FCPE 转换，对比高音段效果
  - ✅ 已验证：FCPE 在 2:17 真假声切换处明显优于 RMVPE
  - ✅ 最优参数：`threshold=0.003, decoder_mode="local_argmax", interp_uv=True`
- [ ] **升级重采样为 SoXR**：`pip install soxr`，修改 mix.py 的重采样逻辑
  - 替换 `scipy.signal.resample` → `soxr.resample(audio, old_sr, new_sr, quality='VHQ')`
  - 消除 Gibbs ringing 伪影
- [ ] **升级分离模型为 BS-RoFormer-Viperx v1297**（SDR 12.97 > 当前 12.96）
  - `audio-separator` 本身活跃（2026-01 更新），换用更高 SDR 模型即可提速无需换引擎
  - 同时测试 `--use_autocast` 等加速参数，看 M5 上能否突破 1:1 实时瓶颈
- [ ] **更新 pyproject.toml 依赖**：添加 soxr、torchfcpe；更新 faiss-cpu 版本约束

### 阶段二：新歌源测试

- [ ] 用户提供新的源歌曲 URL（清亮唱法，非沙哑撕裂版）
- [ ] 用升级后的 pipeline 跑完整流程：下载 → 分离 → 转换 → 混音
- [ ] 对比 V1 的《海阔天空》沙哑版，验证输入源对效果的影响
- [ ] 输出到 iCloud Drive 试听

### 阶段三：调音效果（语音转换参数调优）

- [ ] **transpose 对比**：同一首歌分别跑 transpose 0 / -2 / -3 / -4，输出 4 个版本对比
- [ ] **换声音模型**：从 HuggingFace 下载 2-3 个不同音色的模型试听
  - 优先找中音区饱满的女声模型（如 Adele、Taylor Swift）
  - 优先找有 .index 文件的模型（配合 faiss 启用后效果更好）
- [ ] **index_rate 对比**（如 faiss 已修复）：同一模型分别跑 index_rate 0 / 0.3 / 0.5 / 0.7 对比
- [ ] **F0 方法对比**：~~同一段分别用 RMVPE vs FCPE 对比~~ ✅ 已完成，FCPE 胜出
  - FCPE 最优参数：`threshold=0.003, decoder_mode="local_argmax", interp_uv=True`
  - 12 组参数网格测试，dmloc 系列整体优于 dmarg，th0.003 + iv1 为理论最优且用户确认
- [ ] 所有对比版本命名清晰，输出到 iCloud Drive 让用户逐个试听筛选

### 阶段四：确认最优参数组合

- [x] 根据试听结果，确定最优的 {输入源类型, 模型, transpose, index_rate, f0_method} 组合
- [x] 将最优参数记录到 TODO.md 作为后续批量处理的默认值
- [x] 用最优参数重跑完整歌曲，输出最终成品
- [x] **三轨混音**：转换后主唱(-16 LUFS) + 原伴唱(-22 LUFS) + 伴奏(-18 LUFS)

> **V1.1 最终成品**：`海阔天空_v1.1_三轨混音.wav`（iCloud Drive）
> - 主唱：Ariana Grande RVC v1 + FCPE（th0.003, dmloc, iv1）+ 短分段（x_center=15）
> - 伴唱：原声保留（mel_band_roformer_karaoke 分离）
> - 伴奏：BS-RoFormer ep368 分离
> - 混音：高通 80Hz + 混响（room=0.3, wet=0.15）+ LUFS 归一化 + 防削波
> - **待微调**：主唱音量略大，后续可降低 vocal_lufs 或增加 vocal_gain_db 负值

> **当前已确认参数**：
> - F0 提取器：**FCPE**（`threshold=0.003, decoder_mode="local_argmax", interp_uv=True`）
> - 分段配置：**x_center=15, x_pad=2, x_max=20**（解决长序列 Hubert 注意力退化）
> - SoXR VHQ 重采样 ✅
> - faiss subprocess 隔离 + `KMP_DUPLICATE_LIB_OK=TRUE` ✅

---

## V1.2：多模型对比测试 ✅ 已完成

> 目标：用多个声音模型转换同一首歌（《海阔天空》原版），A/B 试听找到最佳声音。
> **状态：已完成** — Hayley Williams + J 版本效果链产出最终三轨混音。

### 阶段一：模型收集 ✅

- [x] 清理上次下载失败的空目录（adele, amy_winehouse, chris_cornell, freddie_mercury, hozier, the_weeknd 等 12 个）
- [x] 从 HuggingFace `binant` 用户仓库下载 4 个新模型（直接 .pth + .index，非 zip）
- [x] 验证所有 11 个模型可加载，检测 RVC 版本

> **模型清单**：
>
> | 模型 | RVC 版本 | 来源 | 类型 | 特点 |
> |------|---------|------|------|------|
> | Adele | v2 | binant/Adele__RVC_-_400_Epochs_ | 🆕 | 女声，力量型 |
> | Ariana Grande | v1 | V1 遗留 | 旧 | 女高音 |
> | Billie Eilish | v2 | V1 遗留 | 旧 | 女声，低沉 |
> | Chester Bennington | v2 | binant/Chester_Bennington__RVC__1000_epochs | 🆕 | 男声，沙哑摇滚 |
> | Dua Lipa | v2 | Roscall/RVCModels | 旧 | 女流行 |
> | Freddie Mercury | v2 | binant/Freddie_Mercury__RVC_-_700_Epochs_ | 🆕 | 男声，传奇高音 |
> | Jay Chou 周杰伦 | v2 | V1 遗留 | 旧 | 华语男声 |
> | Kurt Cobain | v2 | binant/RVC-Models | 旧 | 男声，沙哑摇滚 |
> | Michael Jackson | v1 | V1 遗留 | 旧 | 男流行 |
> | Taylor Swift | v2 | binant/RVC-Models | 旧 | 女流行 |
> | The Weeknd | v2 | binant/The_Weeknd__RVC__1000_Epochs | 🆕 | 男声，R&B |
>
> **下载经验**：
> - HuggingFace 大文件并行下载容易截断，必须逐个串行下载 + `--retry 3`
> - `binant` 用户仓库最可靠：根目录直接放 `model.pth` + `model.index`，无 zip 嵌套
> - 大型合集仓库（juuxn/RVCModels 等）全是 zip/UUID 命名，不可用
> - librarian 给的 URL 全部 404/401，必须用 HuggingFace API 验证后再下载

### 阶段二：批量转换 ✅

- [x] 验证 FCPE monkey-patch 仍存在于 pipeline.py（完好）
- [x] 用 11 个模型逐一转换《海阔天空》主唱（lead vocals）
- [x] 所有转换版本输出到 iCloud Drive `ai_song_output/v1.2/`

> **批量转换结果**（2026-03-12）：
>
> | 模型 | 耗时 | 输出大小 | 状态 |
> |------|------|---------|------|
> | Adele | 41.4s | 19.9MB | ✅ |
> | Ariana Grande | 38.6s | 24.9MB | ✅ |
> | Billie Eilish | 37.9s | 24.9MB | ✅ |
> | Chester Bennington | 38.6s | 24.9MB | ✅ |
> | Dua Lipa | 38.8s | 24.9MB | ✅ |
> | Freddie Mercury | 40.0s | 24.9MB | ✅ |
> | Jay Chou | 48.3s | 29.8MB | ✅ |
> | Kurt Cobain | 35.8s | 19.9MB | ✅ |
> | Michael Jackson | 38.1s | 24.9MB | ✅ |
> | Taylor Swift | 47.6s | 29.8MB | ✅ |
> | The Weeknd | 39.1s | 24.9MB | ✅ |
>
> **总耗时 444 秒（7 分 24 秒），11/11 全部成功。**
> 转换参数：FCPE（th0.003, dmloc, iv1）+ 短分段（x_center=15）+ transpose=0 + index_rate=0.0
> 输出为纯人声转换（不含伴奏混音），便于直接对比声音质量。

### 阶段三：用户试听 & 选择 ✅

> **第一批（11个模型）反馈**：音域窄、高音不稳、音色不适合唱歌。用户要求"音域广、声音干净、有辨识度"的歌手模型。

- [x] 用户试听第一批 11 个模型
- [x] F0 音域分析：发现多数模型高音区不稳定，Chester Bennington 和 Jay Chou 高音稳定性最佳
- [x] 用户明确需求："找到一些有特色的好听的声音，去唱一些我熟悉的歌，找点不一样的感觉"

> **第二批（10个高质量模型）下载 & 转换**：
>
> | 模型 | RVC 版本 | Epochs | 来源 | 性别 | 特点 |
> |------|---------|--------|------|------|------|
> | Whitney Houston | v2 | 1000 | QuickWick | 女 | 殿堂级嗓音，超宽音域 |
> | Ariana Grande v2 | v2 | 4000 | QuickWick | 女 | 海豚音，高音域（比第一批版本高很多 epochs） |
> | Christina Aguilera | v2 | 800 | QuickWick | 女 | 力量型，转音华丽 |
> | Beyonce | v2 | 1000 | QuickWick | 女 | 音色丰富，节奏感强 |
> | Celine Dion | v2 | 1200 | mrkmja/Celine2020s | 女 | 经典高音，情感表达力强 |
> | Hayley Williams | v2 | 600 | QuickWick | 女 | 摇滚嗓，穿透力强 |
> | YOASOBI (几田莉拉) | v2 | 1000 | QuickWick | 女 | 日系清亮音色，辨识度高 |
> | Ed Sheeran | v2 | 1000 | QuickWick | 男 | 温暖偏沙哑，叙事感强 |
> | Kenshi Yonezu (米津玄師) | v2 | 1000 | QuickWick | 男 | 日系独特音色，真假声切换自然 |
> | Chris Martin (Coldplay) | v2 | 1000 | QuickWick | 男 | 假声出色，空灵感 |
>
> **模型来源**：`QuickWick/Music-AI-Voices`（HuggingFace，877 个模型目录）+ `mrkmja` 个人仓库
> **10/10 全部转换成功**，总耗时 345 秒。输出到 iCloud `ai_song_output/v1.2_new/`

- [x] 用户试听第二批，Hayley Williams 反馈最好："力量感不错"
- [x] 用户反馈两个问题：① 声音单薄干涩 ② 元音长音处有抖动

### 阶段四：混音效果链升级 & 转换参数调优 ✅

> **问题诊断**：
> - "单薄干涩"原因：旧效果链只有 HPF 80Hz + 极轻混响（wet=0.15, room=0.3），缺少压缩器/饱和/EQ
> - "抖动"原因：原唱黄家驹的颤音风格被 RVC 忠实还原，需要 F0 平滑
> - "高音失真"原因：效果链中 Distortion drive 过高 + 无 Limiter 保护

#### 混音效果链升级（mix.py `_apply_vocal_effects`）

- [x] 升级效果链：HPF → Compressor → EQ(body/presence/air) → Warmth/Saturation → Delay → Reverb
- [x] A/B 对比确认新效果链明显优于旧版

> **当前最优效果链参数（J 版本，目前最佳效果）**：
>
> | 效果 | 参数 | 说明 |
> |------|------|------|
> | Pre-Gain | -3.0 dB | 预衰减，防止 RVC 输出过热导致后续效果失真 |
> | HPF | 80 Hz | 去低频噪音 |
> | Compressor | threshold=-16dB, ratio=2.5:1, attack=15ms, release=120ms | 温和压缩，让动态稳定 |
> | Makeup Gain | +1.5 dB | 补偿压缩损失 |
> | EQ Body | Low Shelf 250Hz +2.0dB | 增加"身体感" |
> | EQ Presence | Peak 3kHz +1.0dB, Q=1.0 | 增加存在感 |
> | EQ Air | High Shelf 10kHz +2.0dB | 增加"空气感" |
> | Warmth | Distortion drive=1.5dB | 低 drive 软饱和，添加谐波温暖感 |
> | Gain Comp | -0.9 dB | 补偿 Distortion 增益 |
> | Limiter | threshold=-3.0dB, release=50ms | **防止高音过载**，关键！ |
> | Delay | 80ms, mix=8%, feedback=15% | 微妙空间层次 |
> | Reverb | room=0.45, wet=0.22, damping=0.6, width=1.0 | 比旧版更饱满的空间感 |

#### 转换参数调优

- [x] F0 平滑：在 RVC pipeline.py 中注入 Savitzky-Golay 滤波器（window=7, polyorder=2），减弱长元音抖动
- [x] 实验 protect=0.20 + index_rate=0.5 → 高音段失真加重（Hayley 无 index 文件，index_rate>0 有害）
- [x] 最终回退到 protect=0.33 + index_rate=0.0 + F0 平滑 → **J 版本，目前最佳效果**

> **当前最优转换参数**：
>
> | 参数 | 值 | 说明 |
> |------|-----|------|
> | f0_method | fcpe | threshold=0.003, decoder_mode=local_argmax, interp_uv=True |
> | F0 平滑 | Savitzky-Golay (window=7, polyorder=2) | 仅对 voiced 段平滑，减弱颤音抖动 |
> | protect | 0.33 | 辅音保护（降低会导致高音段质量下降） |
> | index_rate | 0.0 | 无 index 文件的模型必须设为 0 |
> | rms_mix_rate | 1.0 | 使用原始音量包络 |
> | filter_radius | 3 | 中位值滤波半径 |
> | transpose | 0 | 不变调 |
>
> **关键发现**：
> - 对没有 .index 文件的模型，index_rate 必须为 0，否则高音段严重失真
> - protect 降低到 0.20 虽改善咬字但恶化高音，0.33 是当前最佳平衡点
> - F0 平滑有效减弱原唱颤音对转换的影响，但不破坏旋律轮廓
> - 效果链中 Distortion drive 不能超过 2dB，否则高音过载；必须搭配 Limiter

#### A/B 对比版本记录

> 所有版本位于 `output/v1.2_new/ab_compare/`，iCloud `ai_song_output/v1.2_new/ab_compare/`
>
> | 版本 | 转换参数 | 混音效果 | 评价 |
> |------|----------|----------|------|
> | A | protect=0.33, idx=0.0, 无F0平滑 | 旧（HPF+轻混响） | 基线，单薄干涩 |
> | B | 同A | 新（压缩+EQ+饱和+延迟+混响） | 比A好，但咬字失真+抖动 |
> | D | protect=0.20, idx=0.5, F0平滑 | 新 | 前半段好，高音严重失真 |
> | G | 同D | 柔和饱和+Limiter | 高音仍差 |
> | **J** | **protect=0.33, idx=0.0, F0平滑** | **预衰减+柔和效果+Limiter** | **✅ 目前最佳** |

### 阶段五：最终成品 ✅

- [x] 更新 MixConfig 默认值匹配 J 版本参数（pre-gain、limiter、调优后的 compressor/EQ/warmth）
- [x] 用最优参数（J 版本配置）对 Hayley Williams 产出完整歌曲三轨混音
- [x] 输出到 iCloud Drive

> **V1.2 最终成品**：`海阔天空_v1.2_hayley_williams_三轨混音.wav`（iCloud Drive）
> - 主唱：Hayley Williams RVC v2 + FCPE + F0 Savitzky-Golay 平滑 + J 版本效果链
> - 伴唱：原声保留（mel_band_roformer_karaoke 分离）
> - 伴奏：BS-RoFormer ep368 分离
> - 效果链：Pre-Gain -3dB → HPF 80Hz → Compressor(-16dB/2.5:1) → EQ(body+presence+air) → Distortion 1.5dB → Limiter -3dB → Delay 80ms → Reverb(room=0.45/wet=0.22)
> - LUFS：Lead -17 / Backing -22 / Instrumental -18 → Mixed -14.1 LUFS

---

## 效果优化（归档 — 已合并到 V1.1）

> 以下方案已整合到 V1.1 计划中，此处保留作为参考。

<details>
<summary>点击展开原始方案列表</summary>

### 改善方案（按优先级）

> V1 pipeline 已跑通但效果需优化。以下为按优先级排列的改善方案。

#### 🔴 高优先级（最可能见效）

- [ ] **换清亮唱法的输入源**：当前沙哑撕裂唱法导致 F0 断裂和 Hubert 特征劣化，是高音断音的最大嫌疑。找一个清亮唱法的翻唱对比，排除输入源问题
- [ ] **修复 faiss，启用 index_rate**：faiss-cpu v1.14.1（2026.3 发布）包含 macOS ARM64 专项修复（#4789, #4755, #4798）。尝试 `pip install faiss-cpu==1.14.1` 或 `conda install -c conda-forge faiss-cpu=1.14.1`，启用 `index_rate=0.7` 提升 timbre 还原度
  - 备选：用 hnswlib 替代 faiss 做最近邻搜索（需改 `vc_infer_pipeline.py`）
- [ ] **升级 F0 提取方法为 FCPE**：FCPE（TorchFCPE）具备上下文感知能力，能显著减少呼吸音和转音处的音高跳变断裂，RPA 96.79% 优于 RMVPE ~95%。`pip install torchfcpe`

#### 🟡 中优先级（调参优化）

- [ ] **换模型**：下载 Adele / Taylor Swift 等中音区饱满的歌手模型对比效果。或寻找华语歌手模型（mxgf.cc / 百度网盘）
- [ ] **降调 transpose -2~-4**：让高音落回模型甜区，各跑一版对比
- [ ] **混合变调**：`--transpose -3 --instrumental-shift -1`，总降4半音但各自在安全范围

#### 🔵 远期（V2）

- [ ] Bounce-Back 变调技法：先 DSP pitch shift 到甜区 → RVC transpose=0（最高保真）→ shift 回原调
- [ ] 升级重采样为 SoXR（当前 scipy.signal.resample 基于 Fourier 变换，易产生 Gibbs ringing；SoXR VHQ 模式质量更高）：`pip install soxr`，用 `librosa.resample(..., res_type='soxr_vhq')`

</details>

---

## V1.5：性能优化（归档 — 分离加速部分已合并到 V1.1）

- [ ] 将 `audio-separator` 切换为 `mlx-audio-separator`（MLX 原生，M4 Max 上 ~34x 实时；M5 预计更快）
  - 或切换为 `demucs-mlx`（v1.4.3，M4 Max 上 73x 实时，7 分钟歌曲 ~12 秒分离）
- [ ] 验证模型兼容性（BS-RoFormer 等模型在 MLX 版本下效果一致）

---

## V2：伴唱处理 & 精细化 & 自动评估

- [x] 引入 MelBand RoFormer Karaoke 模型，实现主唱/伴唱分离
- [ ] 分离后主唱单独做 voice conversion，伴唱保留原声或另行处理
- [ ] 混音阶段适配三轨合并（转换后主唱 + 原伴唱 + 伴奏）
- [ ] **自动化伴唱检测**：分离后检测伴唱文件 RMS/dBFS，低于阈值（如 -40dBFS）则判定"无伴唱"，跳过伴唱处理
  - 方案 A（推荐）：先分离再判断 — 检测伴唱能量，低于阈值跳过后续处理，逻辑最简单
  - 方案 B：分离前预检测 — 用频谱分析/声道相关性判断是否有多声部，省 3 分钟但实现复杂
- [ ] "Bounce-Back" 变调技法：
  1. 用 pedalboard 将源人声 pitch shift 到模型甜区
  2. RVC 推理 transpose = 0（模型最高保真）
  3. 将转换后音频 shift 回原调
  - 让 RVC 永远在甜区工作，大跨度 shift 交给 DSP 工具处理
- [ ] 自动质量评估（用于批量模型试听场景，自动排序后人工只听 top N）：
  - SingMOS-Pro：专为歌声训练的 MOS 预测，给"好听度"分数（2025 年末发布）
  - F0 曲线对比（FCPE / torchcrepe）：检测音准偏差，跑调是最致命问题
  - UTMOSv2（可选）：对合成感/金属感/爆音等瑕疵敏感，快速筛掉明显有问题的
  - MCD（Mel Cepstral Distortion）：频谱距离评估原始声音与合成声音相似度
- [ ] 混音升级：引入 Matchering 2.0 做参考曲目自动母带处理
- [ ] 考虑替换 RVC → **GPT-SoVITS** / **DDSP-SVC**（无 faiss 依赖，活跃维护，更好的高音处理）

---

## V3：换代升级（远期）

- [ ] 评估 **GPT-SoVITS**（55.7k ⭐，2026-02 活跃）：零样本转换（5 秒参考音），无 fairseq/faiss 依赖，MPS 稳定
- [ ] 评估 **MLX-Audio**（6.2k ⭐，2026-03 活跃）：MLX 原生语音转换，M5 上 3x 快于 PyTorch，无 faiss 依赖
- [ ] 评估 **DDSP-SVC**（2.5k ⭐，2026-02 活跃）：微分信号处理合成器，完美保留共振峰（Formant），轻量低功耗
- [ ] 下载工具：评估 **BBDown**（13.5k ⭐，2026-01 活跃）B站专用，Hi-Res FLAC 无损音频流
- [ ] ~~Seed-VC~~（已归档，不再考虑）

---

## 工具选型备忘（2026.3 调研）

> 数据来源：GitHub API，查询时间 2026-03-11。
> 维护状态标准：✅ 活跃 = 6 个月内有更新 | ⚠️ 缓慢 = 6-12 个月 | ❌ 停滞/归档 = 超过 12 个月

### 全部工具维护状态一览

| 工具 | 仓库 | 最后更新 | Stars | 归档 | 维护状态 |
|------|------|---------|-------|------|---------|
| **yt-dlp** | yt-dlp/yt-dlp | 2026-03-11 | 150k | 否 | ✅ 非常活跃 |
| **BBDown** | nilaoda/BBDown | 2026-01-10 | 13.5k | 否 | ✅ 活跃 |
| **bilibili-api** | Nemo2011/bilibili-api | 2026-02-19 | 3.6k | 否 | ✅ 活跃 |
| **audio-separator** | nomadkaraoke/python-audio-separator | 2026-01-24 | 1.1k | 否 | ✅ 活跃 |
| **demucs** | facebookresearch/demucs | 2024-04-24 | 9.8k | **是** | ❌ **已归档** |
| **demucs-mlx** | ssmall256/demucs-mlx | 2026-03-06 | 5 | 否 | ⚠️ 极早期（5 星） |
| **mlx-audio-separator** | ssmall256/mlx-audio-separator | 2026-03-06 | 0 | 否 | ⚠️ 极早期（0 星） |
| **rvc-python** | daswer123/rvc-python | 2024-10-18 | 142 | 否 | ❌ **停滞 17 个月** |
| **RVC WebUI** | RVC-Project/...WebUI | 2024-11-24 | 34.7k | 否 | ❌ **停滞 16 个月** |
| **Seed-VC** | Plachtaa/seed-vc | 2025-04-20 | 3.6k | **是** | ❌ **已归档** |
| **DDSP-SVC** | yxlllc/DDSP-SVC | 2026-02-22 | 2.5k | 否 | ✅ 活跃 |
| **GPT-SoVITS** | RVC-Boss/GPT-SoVITS | 2026-02-09 | 55.7k | 否 | ✅ **非常活跃** |
| **MLX-Audio** | Blaizzy/mlx-audio | 2026-03-10 | 6.2k | 否 | ✅ **非常活跃** |
| **OpenVoice** | myshell-ai/OpenVoice | 2025-04-19 | 36k | 否 | ⚠️ 缓慢（11 个月） |
| **pedalboard** | spotify/pedalboard | 2026-02-02 | 6k | 否 | ✅ 活跃 |
| **pyloudnorm** | csteinmetz1/pyloudnorm | 2026-01-04 | 760 | 否 | ✅ 活跃 |
| **matchering** | sergree/matchering | 2025-11-26 | 2.4k | 否 | ✅ 活跃 |
| **faiss** | facebookresearch/faiss | 2026-03-10 | 39.3k | 否 | ✅ 非常活跃 |
| **hnswlib** | nmslib/hnswlib | 2025-09-14 | 5.1k | 否 | ⚠️ 缓慢（6 个月） |
| **TorchFCPE** | CNChTu/FCPE | 2025-10-14 | 189 | 否 | ⚠️ 缓慢（5 个月） |
| **torchcrepe** | maxrmorrison/torchcrepe | 2025-05-16 | 508 | 否 | ⚠️ 缓慢（10 个月） |
| **python-soxr** | dofuuz/python-soxr | 2025-10-12 | 107 | 否 | ⚠️ 缓慢（5 个月） |

### 关键发现

**❌ 已死/不可用：**
- **demucs** — Meta 已归档，不再维护
- **rvc-python** — 停滞 17 个月，依赖全部过时，Apple Silicon 上 faiss SEGFAULT
- **RVC WebUI** — 停滞 16 个月，34.7k 星但无人维护
- **Seed-VC** — 已归档，虽然技术先进但无法获得更新

**⚠️ 谨慎使用：**
- **demucs-mlx / mlx-audio-separator** — 代码活跃但 0-5 星，个人项目，风险高
- **OpenVoice** — 11 个月无更新，可能进入维护模式
- **TorchFCPE / torchcrepe / python-soxr** — 工具库性质，更新慢但功能稳定，可用

**✅ 推荐使用：**
- **GPT-SoVITS** — 55.7k 星，2 月还在更新，社区最活跃的语音转换项目
- **DDSP-SVC** — 2.5k 星，2 月更新，无 faiss，MPS 支持
- **MLX-Audio** — 6.2k 星，昨天还在更新，MLX 原生，Apple Silicon 首选
- **audio-separator** — 1.1k 星，1 月更新，当前分离引擎可继续用
- **pedalboard** — Spotify 官方维护，2 月更新，混音效果链稳定可靠
- **faiss** — 39.3k 星，昨天更新，v1.14.1 有 ARM64 修复值得尝试

### 当前使用 vs 推荐升级

| 步骤 | 当前工具 | 维护状态 | 推荐升级 | 升级理由 |
|------|----------|---------|----------|----------|
| **下载** | yt-dlp | ✅ 今天更新 | 暂不换（BBDown 可选） | yt-dlp 极度活跃，BBDown 仅 B站场景更优 |
| **分离** | audio-separator | ✅ 1 月更新 | 暂不换 | demucs-mlx 太早期（5 星），audio-separator 够用 |
| **F0 提取** | RMVPE | ✅ RVC 内置 | FCPE (TorchFCPE) | 5 个月无更新但功能稳定，精度更高 |
| **语音转换** | rvc-python | ❌ 停滞 17 月 | **GPT-SoVITS** 或 **DDSP-SVC** | 当前工具已死，这两个活跃且无 faiss |
| **变调** | pedalboard | ✅ 2 月更新 | 暂不换 | Spotify 官方维护，质量好 |
| **重采样** | scipy.signal | ✅ scipy 活跃 | python-soxr | 质量更高，5 个月无更新但稳定 |
| **响度** | pyloudnorm | ✅ 1 月更新 | 暂不换 | 够用 |
| **混音** | pedalboard | ✅ 2 月更新 | + matchering 做母带 | matchering 去年 11 月更新，稳定 |
| **评估** | 无 | — | SingMOS-Pro（待评估） | 需确认是否有可用 Python 包 |

### 语音转换工具对比（含维护状态）

| 工具 | 最后更新 | Stars | faiss | Apple Silicon | 高音 | 零样本 | 推荐 |
|------|---------|-------|-------|-------------|------|--------|------|
| rvc-python | ❌ 2024-10 | 142 | 是 | ⚠️ 问题多 | 一般 | 否 | ❌ 弃用 |
| RVC WebUI | ❌ 2024-11 | 34.7k | 是 | ⚠️ 有修复 | 一般 | 否 | ❌ 停滞 |
| Seed-VC | ❌ 归档 | 3.6k | 否 | ✅ MPS | 优秀 | 是 | ❌ 已死 |
| DDSP-SVC | ✅ 2026-02 | 2.5k | 否 | ✅ MPS | 优秀 | 否 | ✅ 推荐 |
| **GPT-SoVITS** | ✅ 2026-02 | **55.7k** | **否** | ✅ MPS | 良好 | **是** | ✅ **首选** |
| **MLX-Audio** | ✅ 2026-03 | **6.2k** | **否** | ✅ **MLX** | 优秀 | **是** | ✅ **首选** |
| OpenVoice | ⚠️ 2025-04 | 36k | 否 | ✅ MPS | 一般 | 是 | ⚠️ 观望 |

### 分离模型 SDR 排行（MVSEP 榜单）

| 模型 | SDR (Vocals) | 说明 |
|------|-------------|------|
| BS-RoFormer (MVSEP 2025.7) | **14.58** | 最高分，非公开模型 |
| BS-RoFormer-Viperx v1297 | **12.97** | audio-separator 可用 |
| BS-RoFormer ep368 (当前) | **12.96** | 我们正在使用的 |
| Mel-RoFormer Karaoke (aufr33) | **10.20** | 主唱/伴唱分离专用，**已验证效果良好** |
| HTDemucs v4 | ~8.5 | 四轨分离，demucs 已归档 |

### F0 提取方法对比

| 方法 | RPA 精度 | 速度 | 抗噪 | 上下文感知 | MPS 支持 |
|------|---------|------|------|-----------|---------|
| FCPE | **96.79%** | 极快 | 强 | ✅ 是 | ✅ |
| RMVPE | ~95% | 快 | 强 | ❌ 否 | ✅ |
| CREPE | 高 | 慢 | 弱 | ❌ 否 | ✅ |
| Harvest | 中 | 慢 | 中 | ❌ 否 | ❌ CPU-only |

### 变调策略速查

| 差距 | 策略 | 做法 |
|------|------|------|
| ≤4 半音 | 纯 transpose | RVC transpose 直接搞定 |
| 5~8 半音 | 混合方案 | 伴奏 shift 一部分 + 人声 transpose 一部分，各不超安全范围 |
| >8 半音 | Bounce-Back (V2) | 源人声先 shift 到甜区 → RVC transpose=0 → 结果 shift 回原调 |
