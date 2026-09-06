"""Supervised GNN encoder training (malicious-pseudonym classifier).

Ported from ``code/train/gnn_train.py``. Saves ``best_model.pt`` (state dict),
``scaler.pkl``, ``metrics.json`` and ``curves.json`` into the run dir so the
simulation layer can consume them.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from rlac.models.gnn import GNNModel
from rlac.utils.random import seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default="default_exp")
    ap.add_argument("--use_timestamp", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--clip_norm", type=float, default=2.0)
    ap.add_argument("--input_path", default="data/graphs.pt")
    ap.add_argument("--output_dir", default="outputs/gnn_outputs")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    run_dir = (os.path.join(exp_dir, "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
               if args.use_timestamp else exp_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Run Dir] {run_dir}")

    graphs = torch.load(args.input_path, weights_only=False)
    if len(graphs) < 10:
        raise RuntimeError("too few graphs")
    n, n_val = len(graphs), int(len(graphs) * args.val_ratio)
    n_train = int(len(graphs) * args.train_ratio)
    train_g = graphs[:n_train]
    val_g = graphs[n_train:n_train + n_val]
    test_g = graphs[n_train + n_val:]

    scaler = StandardScaler().fit(np.vstack([g.x.numpy() for g in train_g]))
    joblib.dump(scaler, os.path.join(run_dir, "scaler.pkl"))

    def scale(gs):
        for g in gs:
            g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
        return gs

    train_g, val_g, test_g = scale(train_g), scale(val_g), scale(test_g)
    loaders = [DataLoader(train_g, batch_size=args.batch_size, shuffle=True),
               DataLoader(val_g, batch_size=args.batch_size),
               DataLoader(test_g, batch_size=args.batch_size)]

    labels = torch.cat([g.y for g in train_g]).long()
    counts = torch.bincount(labels, minlength=2).float()
    pos_weight = counts[0] / counts[1].clamp(min=1.0)
    model = GNNModel(in_dim=train_g[0].x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        ts, ps, pr = [], [], []
        for b in loader:
            b = b.to(device)
            logit = model(b.x, b.edge_index, b.edge_attr).squeeze(-1)
            prob = torch.sigmoid(logit)
            ts.append(b.y.cpu()); ps.append((prob > 0.5).long().cpu()); pr.append(prob.cpu())
        yt = torch.cat(ts).numpy(); yp = torch.cat(ps).numpy(); ypr = torch.cat(pr).numpy()
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(yt, ypr)
        except Exception:
            auc = float("nan")
        return f1, auc, yt, yp

    best_f1, best_state, bad = -1.0, None, 0
    curves = {"epoch": [], "train_loss": [], "val_f1": [], "val_auc": []}
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for b in loaders[0]:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(model(b.x, b.edge_index, b.edge_attr).squeeze(-1), b.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            opt.step()
            loss_sum += loss.item()
        val_f1, val_auc, _, _ = evaluate(loaders[1])
        sched.step(val_f1)
        curves["epoch"].append(epoch)
        curves["train_loss"].append(loss_sum)
        curves["val_f1"].append(val_f1)
        curves["val_auc"].append(val_auc)
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {epoch}")
                break

    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))
    model.load_state_dict(best_state)
    test_f1, test_auc, yt, yp = evaluate(loaders[2])
    report = classification_report(yt, yp, digits=4, zero_division=0)
    print(f"\nF1={test_f1:.4f} AUC={test_auc:.4f}\n{report}")
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"test_f1": float(test_f1), "test_auc": float(test_auc),
                   "best_val_f1": float(best_f1), "report": report}, f, indent=2)
    with open(os.path.join(run_dir, "curves.json"), "w") as f:
        json.dump(curves, f, indent=2)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
