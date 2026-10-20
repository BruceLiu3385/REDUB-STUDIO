# Ubuntu / AutoDL 部署

从仓库根目录执行下列命令。默认以 AutoDL 容器内 root 用户部署到 `/root/redub-deploy`，大文件写入 `/root/autodl-tmp`。其他 Ubuntu 主机需要自行准备 NVIDIA 驱动、CUDA 可用的基础 PyTorch、Conda、Git、Python 和 pip；脚本不会管理云实例。已运行实例的推理已验证，仓库整理后的全新实例安装仍需复测。

## 1. 硬件与目录

当前硬件预检要求可用 BF16 的 NVIDIA GPU、至少约 24GB 显存、至少 55 GiB 有效容器内存，以及新装 TBDub 时至少 36 GiB 可用磁盘。换声和分离的独立环境、缓存与输出还需额外空间；换声脚本新建环境时另检查至少 18 GiB 可用空间。以脚本预检结果为准，不要仅看宿主机总内存。

模型环境采用 Python 3.10、Torch 2.9.0 与 CUDA 12.8 构建；Web 环境采用 Python 3.12。各安装器保留环境隔离，不向基础 Jupyter 环境安装模型包。网络需能访问脚本列出的官方代码、包索引及模型源。

[`.env.example`](../.env.example) 给出路径配置。应用不自动加载 `.env`；复制并核对内容后，在当前 Linux 终端手动导出：

```bash
cp .env.example .env
# 先编辑 .env，使路径与当前设备一致。
set -a
source .env
set +a
```

```bash
git clone https://github.com/BruceLiu3385/REDUB-STUDIO.git /root/redub-deploy
cd /root/redub-deploy
bash deploy/tbdub_autodl/00_check_hardware.sh
```

默认路径如下。自定义目录时，在运行安装、启动及 Notebook 前导出同一组变量；TBDub 安装器使用 `TBDUB_*`，Web 使用 `REDUB_TBDUB_*`，两组必须指向同一位置。

| 配置 | 默认值 |
| --- | --- |
| `TBDUB_ROOT` / `REDUB_TBDUB_ROOT` | `/root/autodl-tmp/tbdub` |
| `TBDUB_ENV` / `REDUB_TBDUB_ENV` | `/root/autodl-tmp/tbdub/env` |
| `REDUB_PROJECT_ROOT` | 包含 `apps/`、`deploy/` 的仓库根目录，启动脚本默认自动推导 |
| `REDUB_WEB_ROOT` / `REDUB_WEB_ENV` | `/root/autodl-tmp/redub-web` / 其 `env` |
| `REDUB_VOICE_ROOT` / `REDUB_VOICE_ENV` | `/root/autodl-tmp/redub-voice` / 其 `env` |
| `REDUB_SEPARATOR_ROOT` / `REDUB_SEPARATOR_ENV` | `/root/autodl-tmp/redub-separator` / 其 `env` |
| `REDUB_UPSCALE_MODEL` | `/root/autodl-tmp/redub-upscale/RealESRGAN_x2plus.pth` |
| `REDUB_DATA_DIR` | `/root/autodl-tmp/redub-data` |
| `REDUB_PORT` | `7860`，用于直接启动 `03_start_web.sh` |
| `REDUB_FFMPEG` / `REDUB_FFPROBE` | TBDub 环境内工具，ffprobe 也可回退到 PATH |

例如将 TBDub 放到其他磁盘：

```bash
export TBDUB_ROOT=/data/tbdub
export TBDUB_ENV="$TBDUB_ROOT/env"
export REDUB_TBDUB_ROOT="$TBDUB_ROOT"
export REDUB_TBDUB_ENV="$TBDUB_ENV"
```

## 2. 安装并校验 TBDub Student

```bash
bash deploy/tbdub_autodl/01_install_environment.sh
bash deploy/tbdub_autodl/02_download_student.sh
bash deploy/tbdub_autodl/03_verify_assets.sh
```

部署版本默认固定云端实测 TBDub 代码 `19e18e4ba8e2a19e6716cb38110ee2de1dae9bb3`，下载器只准备 Student 与必需辅助模型，不下载 Teacher 全套。模型来源与固定 revision 见 [来源清单](../deploy/tbdub_autodl/sources.json)。若已存在源码仓库，安装器不会悄悄更新它；核对实际 `logs/code_revision.txt` 和目标版本，避免将未知旧目录当作本次固定版本。

下载记录保存在 `$TBDUB_ROOT/logs/download_receipt.json`，列出模型仓库 revision、文件大小及 SHA-256。校验器检查 Student、VAE、空提示、MediaPipe 人脸关键点和 HuBERT 文件，以及模型配置与 safetensors 头。保留这些记录和 `pip-freeze.txt` 以复现环境；不要手工替换模型而沿用旧校验记录。

可用自己的素材先做单模型样片：

```bash
bash deploy/tbdub_autodl/04_run_sample.sh \
  /absolute/portrait.mp4 /absolute/driving.wav 5 first_sample
```

此脚本仅做 TBDub，最长 15 秒，默认启用 CPU offload，并记录日志、时间与整块可见 GPU 的采样占用。Web 流水线有自己的批处理参数，不应用该脚本耗时推算整个工作台速度。

## 3. Web、媒体与声音环境

```bash
bash deploy/redub_studio/00_install_web.sh
bash deploy/redub_studio/00a_install_media.sh
bash deploy/redub_studio/01_install_voice.sh
```

Web 安装器准备独立 Python 3.12 并安装固定依赖；不会加载模型。TBDub 使用的 imageio-ffmpeg 提供 FFmpeg，但不包含 ffprobe；媒体脚本在缺少 ffprobe 时通过 Ubuntu 包管理器安装 ffmpeg。无 root 权限时自行安装可用工具，显式设置 `REDUB_FFMPEG`、`REDUB_FFPROBE`。

换声安装器固定 Chatterbox 代码和 S3Gen VC 权重，核对完整 SHA-256，只下载实际使用的 VC 主权重，不下载完整 TTS 模型。它有意保留特定 PyTorch、Diffusers 等兼容覆盖，并记录模型加载及 CUDA 检查结果；这些检查不等于中文音色质量通过。

`00b_clone_voice_env.sh` 是已有 Conda 环境的可选迁移辅助工具，默认只检查。标准安装不要求运行它；采用克隆时按其 `--help` 完成独立环境复制、匹配 torchaudio，再运行声音安装器，不能让 Web、换声和 TBDub 共用同一个环境目录。

需要背景音乐时再安装分离器：

```bash
bash deploy/redub_studio/02_install_separator.sh
```

安装期预取官方 htdemucs，并将完整哈希写入分离根目录 `logs/runtime-check.json`。请求阶段只接受已校验本地权重，不允许临时自动下载。删除缓存或记录后应重新运行安装器。

## 4. 可选高清增强

```bash
python3 deploy/redub_studio/07_install_upscale.py --help
python3 deploy/redub_studio/07_install_upscale.py
```

安装器下载官方 RealESRGAN_x2plus 并校验固定 SHA-256。无需安装完整 BasicSR/Real-ESRGAN Python 包；增强 worker 复用 TBDub 环境中的 torch、OpenCV、NumPy。未安装权重时仍可制作原版成片。

可用 `--model /absolute/RealESRGAN_x2plus.pth` 或 `--directory /absolute/model-dir` 指定位置；启动服务时同步导出 `REDUB_UPSCALE_MODEL`。默认下载 socket 超时为 120 秒，可用 `--timeout` 调整；已有文件通过校验时直接复用，校验失败默认保留并报错，`--force` 允许下载校验成功后替换无效文件。不支持用任意同名权重替换官方 x2plus。

## 5. 启动与私有访问

推荐用私有启动器检查并启动后台 Web：

```bash
python3 deploy/redub_studio/06_start_private_web.py
```

启动器以数据目录锁避免重复启动，复用相同 Web 环境、相同数据目录且健康的服务，输出实际端口；默认新启动端口为 7860。日志为 `$REDUB_DATA_DIR/web-start.log`。默认端口被其他程序占用时报告错误，不终止其他程序。启动等待最多约 30 秒，未就绪时稍后重跑。

需要前台调试或显式端口时使用：

```bash
REDUB_PORT=7860 bash deploy/redub_studio/03_start_web.sh
```

两种方式都只监听实例内 `127.0.0.1`。关闭前台服务按 Ctrl+C；后台服务不随 SSH 管理连接关闭。实例关机仍会终止服务和当前计算，重启后需再次启动。

### SSH 转发

在自己的电脑保持以下连接，再打开 `http://127.0.0.1:7860/`：

```bash
ssh -N -L 7860:127.0.0.1:7860 -p <控制台SSH端口> root@<控制台SSH主机>
```

首次连接核对控制台提供的 SSH 主机指纹。密码通过 SSH 交互输入，不放入仓库、命令参数或日志。本机端口被占用时可将左侧端口改为其他空闲端口；若服务实际运行在其他端口，右侧端口也应相应调整。

Windows PowerShell 可在本机仓库目录使用系统 OpenSSH 包装脚本，将示例主机和端口替换为控制台值：

```powershell
.\tools\Start-RedubTunnel.ps1 -SshHost 'your-ssh-host.example' -SshPort 12345
# 可选：-SshUser root -LocalPort 7860 -RemotePort 7860
```

脚本只在本机回环地址转发，正常校验首次主机指纹，不保存密码。终端保持运行；连接断开后重新执行。

### 已认证的 Jupyter 代理

先确定运行 Jupyter Server 的真实 Python 环境，安装到该环境，而非 Web 或模型环境。代理安装器默认只检查版本与 dry-run 计划：

```bash
bash deploy/redub_studio/04_install_proxy.sh --python /actual/jupyter/bin/python
bash deploy/redub_studio/04_install_proxy.sh --python /actual/jupyter/bin/python --install
```

检查后，保存 Notebook 并仅重启已确认的 Jupyter Server 服务；重启 kernel 不足以加载服务器扩展。不要重启全部 supervisor 服务、清空认证或杀掉全部 Python 进程。保留既有 `base_url`、IdentityProvider、cookie secret、XSRF 和上游网关认证。

登录现有 Jupyter 后，在原 base URL 下访问 `proxy/7860/`，末尾保留 `/`。默认 `/jupyter/` 对应 `/jupyter/proxy/7860/`。未登录访问应进入登录流程或被拒绝；已登录应能打开页面、上传和下载。若入口本身无认证，应先核实上游访问控制。

在 Jupyter 打开 `deploy/redub_studio/启动口播工作台.ipynb`，运行所有单元格后点击“进入口播工作台”。内核工作目录须位于仓库或其子目录，Notebook 会向上寻找仓库；也可在启动 Jupyter 前设置 `REDUB_PROJECT_ROOT`。代理前缀默认 `/jupyter/`，其他部署设置 `REDUB_JUPYTER_BASE_URL`，例如根路径 `/`；这些变量必须在 Notebook 内核环境中可见，仅在另一终端导出不会改变已启动内核。

应用没有额外账号与独立 XSRF 校验，所有能访问同一 Jupyter 的人共享同一工作台与任务。前端附带 X-XSRFToken 只是兼容行为，不能据此声称代理已经验证该请求。`05_install_named_web.py` 是历史命名入口实验，不是推荐部署步骤；不要为正常访问执行其 `--install`。

## 6. 功能检查与更新

打开页面后刷新健康状态，先测自己的 5–15 秒短片：原声口型 → 开启换声 → 背景分离 → 高清增强。健康检查只查看工具、模块、固定源码版本和可见 GPU，不能代替模型真实推理。完整 HTTP 测试可使用 [应用说明](../apps/redub_studio/README.md) 中的 `tools/smoke_web.py`。

新任务勾选高清增强，或从最近任务选择已完成成片，点击“增强现有成片”。后者创建独立任务，不重复换声与口型处理，原版保留；取消后重试会重新执行整段超分。

更新前等待队列空闲，备份仓库与数据目录，停止已确认的本项目 Web 进程，再同步代码和重启；不要在处理任务期间直接替换工作文件。任务、上传、权重、缓存、日志和运行验收文件均不应提交 Git。不要将 `ps aux`、`/proc/*/environ` 或 Jupyter 完整启动命令写入共享日志，它们可能带凭据；定位时先用 PID、进程名和已知服务管理器状态确认归属。
