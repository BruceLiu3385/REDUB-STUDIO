# 映声 · 口播工作台

将原视频的口播节奏作为声音来源，使用参考声音转换音色，再用 TBDub 驱动人物动作视频。可选分离并回混背景音乐、增强最终画质。完整安装见 [部署说明](../../docs/deployment.md)，实际验收范围见 [验证记录](../../docs/validation.md)。

## 准备素材

1. 上传带画面与声音的原文案视频。首次使用选 5–15 秒、单人、清楚人声的片段；源口播支持 0.5–600 秒。
2. 上传人物动作视频。脸和嘴清楚、光照稳定、正面或轻微侧面，时长应覆盖口播。开启时长适配时只接受约 3% 的轻微调整，明显过短会拒绝，系统不会循环动作补时长。
3. 开启“使用本人音色”时，上传 5–60 秒参考声音，建议 10–30 秒干净说话音频。模型实际使用前 10 秒作为音色条件，应将有效声音放在开头。
4. 首次关闭背景保留和高清增强，检查声音及口型，再分别开启两项功能对比。
5. 提交后先预听提取原声与转换配音，完成后预览、下载成片。任务串行执行，可取消、失败后重试，并在最近任务中查看结果。

单个文件最多 512 MiB；人物视频最多 660 秒。音轨和画面分别计时，不以容器总时长代替流时长。只使用本人或已获授权的人物、声音和视频素材。

音色转换采用 speech-to-speech，不经过 ASR 后重新 TTS，也不自动翻译。模型可能改变发音、音高和局部停顿；总时长一致不代表每个音素对齐。不换声时可以直接用原口播验证口型功能。背景分离可能残留原人声或损伤音乐，需要试听。

## 高清增强

安装 RealESRGAN_x2plus 后，新任务可勾选“AI 高清增强（最高1080P）”。已有完成任务可点击“增强现有成片”，独立排队增强，无需重新换声或对口型。原任务、原版成片和高清版均保留。

增强保持比例，最多放大 2 倍，以 1080P 尺寸为目标：竖屏 720×1280 通常输出 1080×1920；450×660 输出 900×1320；已有 1080P 或更大画面保持尺寸增强。处理同一条 25 fps 时间轴，校验帧数和帧间距，直接复制音轨，并核对压缩音频哈希及包数。

TBDub 在 512×512 对齐人脸上生成后回贴，清晰人物原素材依然重要。当前采用保守超分，将 70% AI 结果与 30% Lanczos 参考混合，没有额外启用扩散重绘或 GFPGAN 人脸修复。增强会估计细节，不能恢复真实纹理；对比脸部、牙齿、唇缘、头发与连续帧，检查是否闪烁或过锐。

高清阶段可取消、重试。取消后复用已有原版成片，但整段超分重新运行，不支持逐帧断点续跑。TBDub 子进程退出后再运行高清 worker，避免两套模型同时常驻显存。

## 代码与运行方式

| 模块 | 职责 |
| --- | --- |
| `redub_studio/server.py` | 上传、任务状态、队列、下载与能力检查 |
| `redub_studio/media.py` | FFmpeg/ffprobe、时间轴、分段、取消与媒体校验 |
| `redub_studio/pipeline.py` | 提取、分离、换声、分段口型、拼接及增强编排 |
| `redub_studio/workers/` | 独立环境运行的 VC、Demucs、TBDub、超分适配器 |
| `redub_studio/static/` | 无构建步骤的 HTML、CSS 与 JavaScript 界面 |
| `tests/` | API、合成音频时间轴、真实 FFmpeg 容器与缓存/取消回归 |

VC 以绝对音频采样位置切窗，携带上下文并交叉淡化；视频默认约 15 秒分段，裁去上下文后在全局 25 fps 时间轴拼接。缓存与完成标记可复用已验证步骤，但这不保证长视频接缝不可察觉。缺少真实模型或校验失败会报告错误，不会用原声假装换声成功。

## Linux / WSL 开发与测试

下面从仓库根目录执行，仅安装 Web 与测试依赖，不会安装或运行 GPU 模型。需要 Python 3.12；媒体测试还需 PATH 中的 `ffmpeg`、`ffprobe`，并支持 libx264 与 AAC 编码。Ubuntu/WSL 可先安装系统媒体工具：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

先准备可用的 Python 3.12 与 venv 支持；Ubuntu 版本未提供时自行安装该解释器。根目录 `pytest.ini` 配置包路径和测试目录；`requirements-dev.txt` 包含声音时间轴回归所需 NumPy，缺失时相关测试会跳过，不能将跳过项记为通过。FFmpeg/ffprobe 缺失也会使真实媒体测试跳过。

在同一终端启动开发服务：

```bash
export REDUB_DATA_DIR="$PWD/.redub-data"
export PYTHONPATH="$PWD/apps/redub_studio"
python -m redub_studio.server
```

访问 `http://127.0.0.1:7860/`。没有模型配置时可检查网页、API 和上传探测，但不能完成真实换声/口型/增强。GPU 全流程使用 Linux CUDA 环境及 [部署说明](../../docs/deployment.md) 中的独立模型环境；本仓库未声明 Windows 原生 GPU 部署已通过。

已安装模型后，真实 HTTP 流程工具接受自行提供的素材：

```bash
python tools/smoke_web.py --help
python tools/smoke_web.py \
  --source /absolute/source.mp4 --portrait /absolute/portrait.mp4 \
  --voice /absolute/reference.wav --receipt /tmp/redub-smoke.json

# 可选：--no-voice-conversion、--keep-background、--enhance-video
```

该工具会上传素材并创建真实任务，消耗实例算力；仅面向本机私有 HTTP 入口。生成的验收文件和上传媒体属于运行数据，不要提交到源码仓库。需要对音色、嘴形及连续画面做人工试听观看。

## 私有访问与任务数据

服务固定监听 `127.0.0.1`，没有独立账号体系或应用层 XSRF 校验。通过已认证的 Jupyter 代理或 SSH 转发使用；所有能访问同一 Jupyter 的人可访问同一工作台任务。它适用于单用户私有测试，不能直接作为公网多用户服务。

前端兼容 Jupyter 代理路径，并在修改请求中附带 `_xsrf` cookie 对应的 `X-XSRFToken`。这不代表代理或应用已经完成该请求的 XSRF 验证。不要关闭 Jupyter 登录、XSRF 或跨域保护来解决访问问题。

默认数据目录为 `/root/autodl-tmp/redub-data`，包含上传素材、任务 JSON、日志和输出。目录不随 Git 同步；升级前备份源码和数据，先等待队列空闲。关闭浏览器、Notebook 或 SSH 转发不会关闭实例；实例关机后 Web 与任务执行也会停止，重启后需重新启动服务。

第三方来源和权重版本见 [模型清单](../../deploy/redub_studio/model_sources.json)、[TBDub 清单](../../deploy/tbdub_autodl/sources.json) 与 [许可声明](THIRD_PARTY_LICENSES.txt)。
