#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$TBDUB_ROOT" /
if command -v findmnt >/dev/null; then findmnt -T "$TBDUB_ROOT"; fi
python - "$TBDUB_ROOT" <<'PY'
import json, os, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
mem = {k: int(v.split()[0])*1024 for k, v in
       (line.split(':', 1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())}
free = shutil.disk_usage(root).free

def cgroup_text(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return ''

host_cpu_count = os.cpu_count() or 1
ram_bytes = mem['MemTotal']
cpu_limits = [float(host_cpu_count)]
if hasattr(os, 'sched_getaffinity'):
    cpu_limits.append(float(len(os.sched_getaffinity(0))))
# The mounted cgroup root is normally the container's own group. Support v2
# and the common v1 controller mount names without reporting host resources
# as the instance allocation.
for path in ('/sys/fs/cgroup/memory.max',
             '/sys/fs/cgroup/memory/memory.limit_in_bytes',
             '/sys/fs/cgroup/memory.limit_in_bytes'):
    value = cgroup_text(path)
    if value.isdigit() and int(value) > 0:
        ram_bytes = min(ram_bytes, int(value))
cpu_max = cgroup_text('/sys/fs/cgroup/cpu.max').split()
if len(cpu_max) == 2 and all(value.isdigit() for value in cpu_max):
    quota, period = map(int, cpu_max)
    if quota > 0 and period > 0:
        cpu_limits.append(quota / period)
for directory in ('/sys/fs/cgroup/cpu', '/sys/fs/cgroup/cpu,cpuacct',
                  '/sys/fs/cgroup'):
    quota = cgroup_text(directory + '/cpu.cfs_quota_us')
    period = cgroup_text(directory + '/cpu.cfs_period_us')
    if quota.isdigit() and period.isdigit() and int(quota) > 0 and int(period) > 0:
        cpu_limits.append(int(quota) / int(period))
record = {'host_cpu_count': host_cpu_count,
          'host_ram_gib': round(mem['MemTotal']/2**30, 2),
          'cpu_count': min(cpu_limits), 'ram_gib': round(ram_bytes/2**30, 2),
          'data_free_gib': round(free/2**30, 2)}
try:
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable'
    record.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                  gpu_gib=round(torch.cuda.get_device_properties(0).total_memory/2**30, 2),
                  bf16_supported=torch.cuda.is_bf16_supported())
    assert record['bf16_supported'], 'BF16 is required'
    x = torch.randn((128, 128), device='cuda', dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item(), 'BF16 matrix test failed'
    del x, y
    torch.cuda.empty_cache()
except ImportError as exc:
    raise SystemExit('Base image must contain working CUDA PyTorch: '+str(exc))
print(json.dumps(record, indent=2))
(root/'logs'/'hardware.json').write_text(json.dumps(record, indent=2)+'\n')
assert record['gpu_gib'] >= 23, 'Choose >=24GB VRAM for this initial setup'
assert ram_bytes >= 55*2**30, 'Choose >=64GB effective system RAM'
if not (root/'env').exists():
    assert free >= 36*2**30, 'Need >=36GiB free before environment + Student installation'
PY
