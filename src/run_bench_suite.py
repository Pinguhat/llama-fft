# run_bench_suite.py
import os
import sys
import time
import subprocess
from datetime import datetime

def run_cmd(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode

def main():
    # -------------------------
    # User config 
    # -------------------------
    python_exe = sys.executable  # uses current venv python
    bench_script = r".\bench_all_in_one.py"

    model_path = r"C:/Users/Lukas/Documents/0-UNI/Seminar/Llama2"
    prompts_file = r".\prompts_eval_20_quality_long.txt"
    calib_dir = r".\calib_out_long_L50_T128_steps1000"

    device = "cuda"
    dtype = "float16"
    batch_size = 2
    limit = 20
    max_len = 128

    warmup = 10
    runs = 50

    # for original we use a single B only (B is irrelevant when num_layers=0),
    # but we keep it explicit to avoid confusion
    orig_block_sizes = "128"

    # patched settings
    patched_block_sizes = "64,128,256"
    num_layers_patched = 1

    # caching: for original set 0, for patched set 1
    cache_cfft_orig = 0
    cache_cfft_patched = 1

    # small pause between runs (helps stability)
    pause_seconds = 3

    # -------------------------
    # Output folder
    # -------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), f"bench_runs_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    def make_common_args():
        return [
            python_exe, bench_script,
            "--model_path", model_path,
            "--prompts_file", prompts_file,
            "--limit", str(limit),
            "--max_len", str(max_len),
            "--device", device,
            "--dtype", dtype,
            "--batch_size", str(batch_size),
            "--runs", str(runs),
            "--warmup", str(warmup),
            "--no_generate",
        ]

    def run_setting(tag, extra_args, run_idx):
        csv_out = os.path.join(out_dir, f"{tag}_run{run_idx}.csv")
        json_out = os.path.join(out_dir, f"{tag}_run{run_idx}.json")
        log_out = os.path.join(out_dir, f"{tag}_run{run_idx}.log.txt")

        cmd = make_common_args() + extra_args + ["--csv_out", csv_out, "--json_out", json_out]
        print(f"\n=== Running: {tag} run {run_idx} ===")
        print("Log:", log_out)
        rc = run_cmd(cmd, log_out)
        if rc != 0:
            print(f"ERROR: run failed (rc={rc}). See log: {log_out}")
            return False
        time.sleep(pause_seconds)
        return True

    ok_all = True

    # -------------------------
    # 1) Original baseline (3 runs)
    # -------------------------
    orig_args = [
        "--block_sizes", orig_block_sizes,
        "--num_layers", "0",
        "--cache_cfft", str(cache_cfft_orig),
    ]
    for i in range(1, 4):
        ok_all &= run_setting("orig", orig_args, i)

    # -------------------------
    # 2) Patched L=1, no calib (3 runs)
    # -------------------------
    l1_nocal_args = [
        "--block_sizes", patched_block_sizes,
        "--num_layers", str(num_layers_patched),
        "--cache_cfft", str(cache_cfft_patched),
    ]
    for i in range(1, 4):
        ok_all &= run_setting("L1_nocal", l1_nocal_args, i)

    # -------------------------
    # 3) Patched L=1, with calib (3 runs)
    # -------------------------
    l1_cal_args = [
        "--block_sizes", patched_block_sizes,
        "--num_layers", str(num_layers_patched),
        "--cache_cfft", str(cache_cfft_patched),
        "--calib_dir", calib_dir,
    ]
    for i in range(1, 4):
        ok_all &= run_setting("L1_cal", l1_cal_args, i)

    print("\n==============================")
    print("DONE.")
    print("Output folder:", out_dir)
    print("All runs OK:" , ok_all)
    print("==============================")

if __name__ == "__main__":
    main()
