import argparse
from pathlib import Path
import gc
import math
from typing import List, Tuple

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

# -----------------------------
# Small helpers
# -----------------------------

def pick_input_device(m):
    if hasattr(m, "hf_device_map"):
        d = m.hf_device_map.get("model.embed_tokens", None)
        if d is None:
            d = next(iter(m.hf_device_map.values()))
        if isinstance(d, int):
            return torch.device(f"cuda:{d}")
        return torch.device(str(d))
    return next(m.parameters()).device


def load_texts(path: Path, limit: int) -> List[str]:
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


def pad_batch(items: List[Tuple[torch.Tensor, torch.Tensor]],pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    items: list of (input_ids, attention_mask), both 1D tensors
    Returns:
      input_ids: (B, T)
      attention_mask: (B, T)
      last_idx: (B,)  last non-pad index per sample
    """
    bsz = len(items)
    max_len = max(int(x[0].numel()) for x in items)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    last_idx = torch.zeros((bsz,), dtype=torch.long)

    for i, (ids, attn) in enumerate(items):
        n = int(ids.numel())
        input_ids[i, :n] = ids
        attention_mask[i, :n] = attn
        last_idx[i] = int(attn.sum().item()) - 1

    return input_ids, attention_mask, last_idx


def gather_last_logits(logits: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V)
    last_idx: (B,)
    returns: (B, V) logits at each sample's last token position
    """
    bsz = logits.shape[0]
    idx = last_idx.view(bsz, 1, 1).expand(bsz, 1, logits.shape[-1])
    return logits.gather(dim=1, index=idx).squeeze(1)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    default_prompts = Path(__file__).with_name("prompts_100.txt")
    parser.add_argument("--prompts_file", type=str, default=str(default_prompts))
    parser.add_argument("--limit", type=int, default=50)

    # run multiple block sizes in one go
    parser.add_argument("--block_sizes", type=str, default="64,128,256")
    parser.add_argument("--num_layers_to_patch", type=int, default=1)

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_bias", action="store_true")

    # performance knobs
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument("--out_dir", type=str, default="calib_out")
    parser.add_argument("--seed", type=int, default=0)

    # cache files
    parser.add_argument("--teacher_cache", type=str, default="teacher_last_cache.pt")
    parser.add_argument("--token_cache", type=str, default="tokenized_prompts_cache.pt")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # A bit faster matmul on RTX cards (safe for inference/training in fp16 setups)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(Path(args.prompts_file), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    # ---------------------------------------
    # Tokenize prompts once (performance win)
    # ---------------------------------------
    token_cache_path = Path(args.token_cache)
    if token_cache_path.exists():
        tokenized = torch.load(token_cache_path, map_location="cpu")
        tok_ids = tokenized["input_ids"]
        tok_attn = tokenized["attention_mask"]
    else:
        tok_ids = []
        tok_attn = []
        for t in texts:
            enc = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_len,
            )
            # store 1D tensors
            tok_ids.append(enc["input_ids"][0].cpu())
            tok_attn.append(enc["attention_mask"][0].cpu())
        torch.save({"input_ids": tok_ids, "attention_mask": tok_attn}, token_cache_path)

    # ---------------------------------------
    # Teacher cache: last-token logits (once)
    # ---------------------------------------
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
    cache_path = Path(args.teacher_cache)

    if cache_path.exists():
        teacher_last_cache = torch.load(cache_path, map_location="cpu")  # list of (V,) float32
    else:
        teacher_last_cache = []
        tdev = pick_input_device(teacher)

        with torch.inference_mode():
            for ids, attn in zip(tok_ids, tok_attn):
                # push to teacher device
                input_ids = ids.unsqueeze(0).to(tdev)
                attention_mask = attn.unsqueeze(0).to(tdev)

                logits_t = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
                # last token position (no padding here, still robust)
                last_idx = int(attn.sum().item()) - 1
                teacher_last = logits_t[:, last_idx, :].to(torch.float32).cpu().squeeze(0)  # (V,)
                teacher_last_cache.append(teacher_last)

        torch.save(teacher_last_cache, cache_path)

    # Teacher muss raus, sonst OOM
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Teacher freed.")

    # parse block sizes
    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]
    if not block_sizes:
        raise ValueError("No block sizes specified")

    # ---------------------------------------
    # Calibrate for each block size
    # ---------------------------------------
    for bsz in block_sizes:
        print(f"\n=== Calibrating block_size={bsz} ===")

        print("Loading student (will be patched) model...")
        student = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            max_memory={0: "6GiB", "cpu": "10GiB"},
            offload_folder=f"offload_student_B{bsz}",
            offload_state_dict=True,
        )
        student.eval()

        patch_mlp_with_block_circulant(
            student,
            num_layers_to_patch=args.num_layers_to_patch,
            block_size=bsz,
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

        # build an index list once (round-robin)
        n_prompts = len(tok_ids)
        steps = args.steps
        batch_size = max(1, args.batch_size)

        for step in range(steps):
            # select indices for this batch (round-robin)
            batch_indices = [((step * batch_size + k) % n_prompts) for k in range(batch_size)]

            # prepare padded batch on CPU, then move to student device
            items = [(tok_ids[i], tok_attn[i]) for i in batch_indices]
            input_ids, attention_mask, last_idx = pad_batch(items, pad_id)

            input_ids = input_ids.to(sdev)
            attention_mask = attention_mask.to(sdev)
            last_idx = last_idx.to(sdev)

            logits_s = student(input_ids=input_ids, attention_mask=attention_mask).logits  # (B,T,V)
            student_last = gather_last_logits(logits_s, last_idx).to(torch.float32)  # (B,V)

            # gather teacher cached last logits for the same prompts
            teacher_last = torch.stack([teacher_last_cache[i] for i in batch_indices], dim=0).to(sdev)  # (B,V)

            # KL(teacher || student) on last-token only (fast and stable)
            p_teacher = F.softmax(teacher_last, dim=-1)
            logp_student = F.log_softmax(student_last, dim=-1)
            loss = F.kl_div(logp_student, p_teacher, reduction="batchmean")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            losses.append(float(loss.item()))
            if (step + 1) % 10 == 0:
                avg10 = sum(losses[-10:]) / 10.0
                print(f"Step {step+1:4d}/{steps} | KL(last-token) avg10={avg10:.6f}")

        # Save only BC params (small checkpoint)
        out_path = out_dir / f"bc_calibrated_B{bsz}.pt"
        save_bc_params(student, str(out_path))
        print(f"Saved: {out_path}")

        # Cleanup per B
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
