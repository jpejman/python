# Created: 2025-05-23 13:06 EDT
# Updated: 2025-05-25 21:40 EDT for v.0.0.7.2.01
# Updated: 2025-05-25 21:55 EDT for v.0.0.7.2.02
# Changelog v.0.0.7.2.00:
# - Bumped versioning to start 7.2.x cycle
# - Added support and placeholders for integrating multiple TTS engines (Coqui, Tortoise)
# - Enhanced comments and structure to prepare for multi-engine orchestration
# - Retained GPU diagnostics and logging from v.0.0.7.1.21
# - ✅ Renamed from 'batch_all_vctk_' to 'batch_all_tts_' to support non-VCTK models in future
# Changelog v.0.0.7.2.02:
# - Enhanced logging label: renamed ERROR_LOG to inference_diagnostics.log
# - Expanded speaker list with preferred high-quality voices
# - Improved LONG_TEXT with complex phrasing, numerics, and currency for pronunciation stress test
# - Prepared structure for Tortoise integration while retaining Coqui logic

import subprocess
import time
from pathlib import Path
from datetime import datetime

SCRIPT_VERSION = "batch_all_tts_v.0.0.7.2.02.py"
CREATED_TIMESTAMP = "2025-05-25 21:55 EDT"
print(f"\U0001F4C4 Script Version: {SCRIPT_VERSION} | Created: {CREATED_TIMESTAMP}\n")

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M-%S")
RUN_ID = f"{SCRIPT_VERSION.replace('.py','')}_{TIMESTAMP}"
OUTPUT_DIR = Path.home() / "tts-output" / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging paths
META_CSV = OUTPUT_DIR / "metadata.csv"
FAILED_CSV = OUTPUT_DIR / "failed.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
ERROR_LOG = OUTPUT_DIR / "inference_diagnostics.log"

with META_CSV.open("w") as f:
    f.write("timestamp,speaker_id,duration_secs,gpu_id,script,output_path\n")
with FAILED_CSV.open("w") as f:
    f.write("timestamp,speaker_id,gpu_id,script,error\n")
ERROR_LOG.touch()

# 🔧 Configuration
# For now, single-GPU Coqui only — Tortoise support to be added incrementally
GPUS = [0]
SPEAKERS = ["p225", "p226", "p227", "p228", "p229", "p238", "p254", "p264", "p267", "p312", "p317"]

# Complex stress test with dates, currency, numbers, punctuation and phrasing variety
LONG_TEXT = (
    "On July 4th, 2024, the price was $235,535.35, which is two hundred thirty-five thousand five hundred thirty-five dollars and thirty-five cents. "
    "This test is designed to verify accurate pronunciation of numbers, dates, punctuation, and currency in synthesized speech. "
    "Furthermore, it includes times such as 8:45 p.m., abbreviations like U.S.A., and phrases like 'twenty-three point seven five percent.'"
)

print(f"🙂 Detected GPUs: {len(GPUS)}")
print("torch.cuda.device_count():", subprocess.getoutput("python3 -c 'import torch; print(torch.cuda.device_count())'"))

# Ensure Docker is working before TTS jobs run
def docker_access_check():
    test_cmd = ["docker", "ps"]
    result = subprocess.run(test_cmd, capture_output=True)
    if result.returncode != 0:
        print("\n❌ Docker access failed: permission denied or docker not running.")
        print("   Hint: run 'sudo usermod -aG docker $USER' and 'newgrp docker' or reboot.")
        exit(1)

# Lightweight CUDA+GPU memory test to ensure drivers are OK
def run_gpu_sanity_test():
    print("\n🔎 Running GPU sanity test...")
    sanity_check = subprocess.run(["python3", "gpu_sanity_test.py"], capture_output=True, text=True)
    print(sanity_check.stdout)
    if sanity_check.returncode != 0:
        print("❌ GPU sanity check failed. Exiting.")
        exit(1)

# Dump nvidia-smi stats for visibility
def log_gpu_stats(prefix):
    try:
        nvidia_output = subprocess.check_output(["nvidia-smi"], text=True)
        with ERROR_LOG.open("a") as log:
            log.write(f"[{TIMESTAMP}] {prefix} GPU STATS:\n{nvidia_output}\n")
    except Exception as e:
        with ERROR_LOG.open("a") as log:
            log.write(f"[{TIMESTAMP}] {prefix} Failed to capture nvidia-smi output: {e}\n")

# Core TTS wrapper for a single speaker
# Will support switching TTS engines in future versions
def synthesize_speaker(speaker_id, gpu_id):
    output_file = f"{speaker_id}_{RUN_ID}.wav"
    container_output_path = f"/workspace/{output_file}"
    host_output_path = OUTPUT_DIR / output_file

    print(f"[GPU {gpu_id}] Checking CUDA availability...")
    cuda_test = subprocess.run([
        "python3", "-c",
        "import torch; assert torch.cuda.is_available(); torch.zeros(1).to('cuda')"
    ], capture_output=True)
    if cuda_test.returncode != 0:
        with FAILED_CSV.open("a") as f:
            f.write(f"{TIMESTAMP},{speaker_id},{gpu_id},{SCRIPT_VERSION},CUDA pre-check failed\n")
        with ERROR_LOG.open("a") as log:
            log.write(cuda_test.stderr.decode("utf-8") + "\n")
        print(f"[GPU {gpu_id}] ❌ CUDA check failed for {speaker_id}")
        return

    print(f"[GPU {gpu_id}] Attempt  1  Speaker: {speaker_id}")
    start_time = time.time()
    log_gpu_stats("BEFORE")

    docker_cmd = [
        "docker", "run", "--rm", "--gpus", f"device={gpu_id}",
        "-e", "COQUI_TOS_AGREED=1",
        "-e", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "-v", f"{OUTPUT_DIR}:/workspace",
        "--entrypoint", "bash",
        "coqui-tts:cuda124-working",
        "-c",
        f"echo '[GPU {gpu_id}] Inside container: CUDA status'; \
         python3 -c \"import torch; print('CUDA OK:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(torch.cuda.current_device()))\"; \
         tts --text \"{LONG_TEXT}\" \
             --out_path {container_output_path} \
             --model_name tts_models/en/vctk/vits \
             --speaker_idx {speaker_id} \
             --use_cuda true \
             --device cuda"
    ]

    result = subprocess.run(docker_cmd, capture_output=True)
    end_time = time.time()
    duration = end_time - start_time
    log_gpu_stats("AFTER")

    stdout_log = result.stdout.decode("utf-8", errors="replace").strip()
    stderr_log = result.stderr.decode("utf-8", errors="replace").strip()

    with ERROR_LOG.open("a") as log:
        log.write(f"[{TIMESTAMP}] GPU {gpu_id} Speaker {speaker_id} COMMAND:\n{' '.join(docker_cmd)}\n")
        log.write(f"[{TIMESTAMP}] GPU {gpu_id} Speaker {speaker_id} STDOUT:\n{stdout_log}\n")
        log.write(f"[{TIMESTAMP}] GPU {gpu_id} Speaker {speaker_id} STDERR:\n{stderr_log}\n\n")

    if result.returncode == 0 and host_output_path.exists():
        filesize = host_output_path.stat().st_size
        with META_CSV.open("a") as f:
            f.write(f"{TIMESTAMP},{speaker_id},{duration:.2f},{gpu_id},{SCRIPT_VERSION},{output_file} ({filesize} bytes)\n")
        print(f"[GPU {gpu_id}] ✅ {speaker_id:6} | Time: {duration:6.2f}s | Size: {filesize:7d} bytes")
    else:
        with FAILED_CSV.open("a") as f:
            f.write(f"{TIMESTAMP},{speaker_id},{gpu_id},{SCRIPT_VERSION},{stderr_log}\n")
        print(f"[GPU {gpu_id}] ❌ FAILURE: {speaker_id}\n{stderr_log}")

# Orchestration
start_time_total = time.time()
docker_access_check()
run_gpu_sanity_test()

for speaker in SPEAKERS:
    synthesize_speaker(speaker, GPUS[0])

end_time_total = time.time()
total_duration = int(end_time_total - start_time_total)
failed_count = sum(1 for line in FAILED_CSV.read_text().splitlines()[1:] if line.strip())

with SUMMARY_CSV.open("w") as f:
    f.write("timestamp,total_speakers,failed,total_duration_secs,script\n")
    f.write(f"{TIMESTAMP},{len(SPEAKERS)},{failed_count},{total_duration},{SCRIPT_VERSION}\n")

print(f"✅ All synthesis jobs complete. Summary written to {SUMMARY_CSV}")
print("\nLet me know if you want to test Tortoise integration, longer test sentences, or quality diagnostics next.")
