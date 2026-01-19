import argparse
from pathlib import Path
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from patch_llama_fft import (
    MODEL_PATH,
    patch_mlp_with_block_circulant,
    BlockCirculantLinear,
    save_bc_params,
)
def pick_input_device(m):
    if hasattr(m, "hf_device_map"):
        d = m.hf_device_map.get("model.embed_tokens", None)
        if d is None:
            d = next(iter(m.hf_device_map.values()))
        if isinstance(d, int):
            return torch.device(f"cuda:{d}")
        return torch.device(str(d))
    return next(m.parameters()).device

def load_texts(path: Path, limit: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        texts.append(s)
        if len(texts) >= limit:
            break
    if not texts:
        raise ValueError(f"No prompts loaded from: {path}")
    return texts

def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

def enable_bc_trainable(model: nn.Module, *, train_bias: bool = True) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, BlockCirculantLinear):
            m.c.requires_grad = True
            n += m.c.numel()
            if train_bias and (m.bias is not None):
                m.bias.requires_grad = True
                n += m.bias.numel()
    return n

def iter_trainable_params(model: nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            yield p

def main():
    parser = argparse.ArgumentParser()
    default_prompts = Path(__file__).with_name("prompts_100.txt")
    parser.add_argument("--prompts_file", type=str, default=str(default_prompts))
    parser.add_argument("--limit", type=int, default=50)

    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--num_layers_to_patch", type=int, default=1)

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_bias", action="store_true")

    parser.add_argument("--out", type=str, default="bc_calibrated.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    texts = load_texts(Path(args.prompts_file), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    print("Loading teacher (original) model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        max_memory={0: "6GiB", "cpu": "10GiB"},
        offload_folder="offload_teacher",
        offload_state_dict=True,
    )
    teacher.eval()

    print("Precomputing teacher last-token cache...")
    cache_path = Path("teacher_last_cache.pt")

    if cache_path.exists():
        teacher_last_cache = torch.load(cache_path, map_location="cpu")
    else:
        teacher_last_cache = []
        tdev = pick_input_device(teacher)

        with torch.inference_mode():
            for t in texts:
                inputs = tokenizer(t, return_tensors="pt")
                inputs = {k: v.to(tdev) for k, v in inputs.items()}
                logits_t = teacher(**inputs).logits
                teacher_last = logits_t[:, -1, :].to(torch.float32).cpu()
                teacher_last_cache.append(teacher_last)

        torch.save(teacher_last_cache, cache_path)

    # Teacher muss raus, sonst OOM
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Teacher freed.")


    print("Loading student (will be patched) model...")
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        max_memory={0: "6GiB", "cpu": "10GiB"},
        offload_folder="offload_student",
        offload_state_dict=True,
    )
    student.eval()

    patch_mlp_with_block_circulant(
        student,
        num_layers_to_patch=args.num_layers_to_patch,
        block_size=args.block_size,
    )

    # Freeze everything, unfreeze only BC params
    freeze_all_params(student)
    n_train = enable_bc_trainable(student, train_bias=args.train_bias)
    print(f"Trainable parameters (BC only): {n_train}")

    optim = torch.optim.AdamW(list(iter_trainable_params(student)), lr=args.lr)
    sdev = pick_input_device(student)

    # simple round-robin over prompts
    print("Starting calibration...")
    losses = []
    for step in range(args.steps):
        t = texts[step % len(texts)]

        inputs = tokenizer(t, return_tensors="pt")
        inputs = {k: v.to(sdev) for k, v in inputs.items()}

        logits_s = student(**inputs).logits

        teacher_last = teacher_last_cache[step % len(texts)].to(sdev)  # CPU -> student device
        student_last = logits_s[:, -1, :].to(torch.float32)

        p_teacher = F.softmax(teacher_last, dim=-1)
        logp_student = F.log_softmax(student_last, dim=-1)
        loss = F.kl_div(logp_student, p_teacher, reduction="batchmean")

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        losses.append(float(loss.item()))
        if (step + 1) % 10 == 0:
            avg10 = sum(losses[-10:]) / 10.0
            print(f"Step {step+1:4d}/{args.steps} | KL(last-token) avg10={avg10:.6f}")

    # Save only BC params (small checkpoint)
    save_bc_params(student, args.out)

    # Cleanup
    del teacher
    del student
    gc.collect()

    print("Done.")

if __name__ == "__main__":
    main()
