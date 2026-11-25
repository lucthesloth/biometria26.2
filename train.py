"""Fine-tune the embedding network on the train split, select the checkpoint by validation EER.

    python train.py --dataset fvc2002_db1 --run-name convnext_tiny_hong --seed 2026

Reads protocols/<dataset>/{train,validation}.csv and validation_pairs.csv, plus the processed images.
Writes runs/<run-name>/seed<seed>/{best.pt,last.pt,train_log.csv,run.json}.
Run name defaults to <dataset>_<backbone>_<normalization>; pass --run-name to override.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from fpverif.config import add_config_args, load_config
from fpverif.data import PKBatchSampler, make_torch_dataset
from fpverif.metrics import eer
from fpverif.seed import seed_everything, seed_worker
from model import ArcFaceHead, build_model, count_params, freeze_until, train_mode


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_transform() -> v2.Compose:
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5], [0.5])])


def train_transform(aug: dict, fill: float) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.RandomAffine(degrees=aug["rotation_deg"], translate=(aug["translate"], aug["translate"]), fill=fill),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=tuple(aug["blur_sigma"]))], p=aug["blur_p"]),
        v2.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])


@torch.no_grad()
def embed_frame(model, frame: pd.DataFrame, image_root: Path, device, batch_size: int = 32) -> dict[str, np.ndarray]:
    ds, _ = make_torch_dataset(frame, image_root, eval_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    chunks = [model(x.to(device)).cpu().numpy() for x, _ in loader]
    emb = np.concatenate(chunks) if chunks else np.zeros((0, 1))
    return dict(zip(frame["path"].tolist(), emb))


def pair_scores(emb: dict[str, np.ndarray], pairs: pd.DataFrame) -> np.ndarray:
    a = np.stack([emb[p] for p in pairs["sample_1"]])
    b = np.stack([emb[p] for p in pairs["sample_2"]])
    return np.einsum("ij,ij->i", a, b)


def validation_eer(model, val: pd.DataFrame, val_pairs: pd.DataFrame, image_root: Path, device) -> tuple[float, float]:
    emb = embed_frame(model, val, image_root, device)
    scores = pair_scores(emb, val_pairs)
    gen, imp = scores[val_pairs["type"] == "genuine"], scores[val_pairs["type"] == "impostor"]
    return eer(gen, imp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--protocol-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg["seed"]
    seed_everything(seed, cfg["train"]["deterministic"])
    device = pick_device(cfg["train"]["device"])
    pp, tr, m = cfg["preprocess"], cfg["train"], cfg["model"]

    protocol_dir = Path(args.protocol_dir or f"protocols/{args.dataset}")
    image_root = Path(pp["output_dir"]) / args.dataset / pp["normalization"]
    train_df = _to_processed(pd.read_csv(protocol_dir / "train.csv"))
    val_df = _to_processed(pd.read_csv(protocol_dir / "validation.csv"))
    val_pairs = _pairs_to_processed(pd.read_csv(protocol_dir / "validation_pairs.csv"))

    run_name = args.run_name or f"{args.dataset}_{m['backbone']}_{pp['normalization']}"
    out = Path(tr["runs_dir"]) / run_name / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    train_ds, label_map = make_torch_dataset(train_df, image_root,
                                             train_transform(tr["augment"], float(pp["background_fill"])))
    labels = train_df["finger"].map(label_map).to_numpy()
    sampler = PKBatchSampler(labels, tr["batch_identities"], tr["samples_per_identity"], seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=tr["num_workers"],
                        worker_init_fn=seed_worker, generator=gen, pin_memory=device.type == "cuda",
                        persistent_workers=tr["num_workers"] > 0)

    model = build_model(cfg).to(device)
    frozen = freeze_until(model, tr["freeze_until"])
    head = ArcFaceHead(m["embedding_dim"], len(label_map), **m["arcface"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["epochs"])

    amp = bool(tr.get("amp")) and device.type == "cuda"
    log, best = [], float("inf")
    t0 = time.time()
    for epoch in range(1, tr["epochs"] + 1):
        train_mode(model, frozen)
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                emb = model(x)
            emb = emb.float()
            loss = F.cross_entropy(head(emb, y), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_eer, val_thr = validation_eer(model, val_df, val_pairs, image_root, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_eer": val_eer, "val_eer_threshold": val_thr,
               "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
        log.append(row)
        print(json.dumps(row))
        torch.save(model.state_dict(), out / "last.pt")
        if val_eer < best:
            best = val_eer
            torch.save(model.state_dict(), out / "best.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)

    (out / "run.json").write_text(json.dumps({
        "dataset": args.dataset, "seed": seed, "config": cfg, "best_val_eer": best,
        "n_train_identities": len(label_map), "n_train_images": len(train_df),
        "params_total": count_params(model), "params_trainable": count_params(model, True),
        "device": str(device), "hardware": _hardware(device), "torch": torch.__version__,
        "train_time_s": time.time() - t0,
    }, indent=2, default=str))


def _to_processed(frame: pd.DataFrame) -> pd.DataFrame:
    """Protocol CSVs point at raw files; processed images are <finger>_<impression>.png."""
    return frame.assign(path=frame["finger"].astype(str) + "_" + frame["impression"].astype(str) + ".png")


def _pairs_to_processed(pairs: pd.DataFrame) -> pd.DataFrame:
    def conv(p: str) -> str:
        stem = Path(p).stem
        return f"{stem}.png"
    return pairs.assign(sample_1=pairs["sample_1"].map(conv), sample_2=pairs["sample_2"].map(conv))


def _hardware(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{platform.processor() or platform.machine()} x{torch.get_num_threads()} threads"


if __name__ == "__main__":
    main()
