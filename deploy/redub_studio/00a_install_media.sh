#!/usr/bin/env bash
# AutoDL's imageio FFmpeg does not include ffprobe. Install the Ubuntu media tools
# from the image's configured repositories; existing model Python packages stay intact.
set -Eeuo pipefail
if command -v ffprobe >/dev/null; then
  ffprobe -version | head -n 1
  exit 0
fi
[[ "$(id -u)" == 0 ]] || { printf 'Run as root, or install ffprobe and set REDUB_FFPROBE.\n' >&2; exit 2; }
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes --no-install-recommends ffmpeg
ffprobe -version | head -n 1
