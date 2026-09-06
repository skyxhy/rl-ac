# RL-based Adaptive Access Control against Misbehavior in Vehicular Networks

Reinforcement-learning driven, **pseudonym-level adaptive access control** to defend against
**DoS / Sybil / mixed misbehavior** in vehicular (V2X/ITS) message exchanges.

> Status: cleaned-up research codebase derived from a graduation project.
> Paper route: RL-simulation only (GNN risk representation + DRQN access control vs. an
> adaptive/FSM attacker); the Hyperledger-Fabric execution layer is **not** part of this repo.

## What it does

1. Parse VeReMi-Extension misbehavior traces (`test-go/`, Go) into per-window message CSVs.
2. Build per-window graph snapshots (nodes = sender pseudonyms; edges = shared-receiver
   Jaccard similarity) and train a **TransformerConv GNN** as a supervised malicious-pseudonym
   classifier / frozen risk encoder (`code/train/gnn_train.py`).
3. Run an adaptive **POMDP / DRQN** defender (`code/train/simulation.py`) that, every window,
   assigns each active pseudonym an **access level** (allow / degrade / deny), trading off
   legitimate-traffic utility against attack leakage through a scale-aware Pareto reward (`alpha`).
4. Compare against baselines: Contextual-Bandit (no TD) and fixed GNN-risk thresholds, across
   attack intensities, observation horizons (`sight`) and seeds. Figures in `figure/`.

## Repository layout

```
data_view/
├── code/                 # Python package
│   ├── env_entity/       #   replay environment: data, attacker FSM, graph, risk, policy
│   ├── entity_agent/     #   DRQN agent + Contextual-Bandit agent (+ replay buffers)
│   ├── model/            #   GNN / STGCN / MLP models
│   ├── train/            #   simulation.py (main), supervised trainers
│   ├── data/             #   feature engineering + dataset builders
│   ├── analysis/         #   correlation / PCA / XGBoost feature analyses
│   └── view/             #   paper-figure scripts (styled_view.py)
├── config/               # YAML base configs (train_rl / train_cb / train_th …)
├── figure/               # publication figures (pdf+svg+png)
├── test-go/              # VeReMi trace parser (Go)
├── main.sh               # full experiment-matrix runner
├── environment.yml       # conda env (name: its-grl)
├── requirements.txt      # pip deps
└── *.sh                  # per-attack ingest + GNN build scripts
```

Run artifacts (`outputs/`, `logs/`, raw `*.csv`, `*.pt`, `*.npz`) are **git-ignored** and
regenerated locally.

## Environment

```bash
# conda
conda env create -f environment.yml
conda activate its-grl

# or pip
pip install -r requirements.txt
```

Tested with Python 3.10 + PyTorch 2.x + torch-geometric 2.x.
> On Windows consoles set `PYTHONIOENCODING=utf-8` (the scripts print emoji progress markers
> that GBK consoles cannot encode).

## Reproduce (from raw data)

```bash
# 1) Get the VeReMi Extension dataset and parse one scenario into messages CSV
go run test-go/main.go <VeReMi-dir> <start_time> <end_time>
cp output.csv sybil.csv            # or dos.csv / dosrandom.csv / dosdisruptivesybil.csv …

# 2) Build graph snapshots + train a supervised GNN risk encoder
python -m code.data.build_gnn_dataset --attack_intensity_list 1.0
python -m code.train.gnn_train --exp_name single_sybil_gnn --use_timestamp 0

# 3) Train / test a defender (single config)
python -m code.train.simulation --config config/train_rl.yaml

# or run the whole experiment matrix
bash main.sh
```

### Minimal smoke (fast sanity checks)
```bash
export PYTHONIOENCODING=utf-8
# threshold test mode (8 steps)
python -m code.train.simulation --config config/_smoke_threshold.yaml
# RL train (6 steps, 1 episode)
python -m code.train.simulation --config config/_smoke_rl.yaml
# CB train (6 steps, 1 episode)
python -m code.train.simulation --config config/_smoke_cb.yaml
```
(Smoke configs are temporary; keep them in `config/` or delete after verification.)

## Experiment output convention

Per-run artifacts are written under `outputs/simulation/<mode>/<save_name>/{png,csv,npy}/`
plus `final_agent.pt` checkpoints and rejection/pressure plots. A `save_name` encodes
`attack(a) alpha(al) sight(s) timesteps(ts/t) seed`, see `main.sh`.

## Notes / known issues (fixed in this cleaning)

- Discounting is now consistent: the replay buffer stores **2-step** returns and the DQN update
  bootstraps with `gamma**2`.
- The Contextual-Bandit ablation now regresses on **single-step immediate rewards** (no look-ahead),
  keeping it a genuine "no temporal credit assignment" baseline.
- The attacker's rejection-rate observation is computed from **actual admitted weight `1-w`**, so it
  is consistent for both 2-level and 3-level action spaces.

## License

MIT — see `LICENSE`. Data: use the public VeReMi-Extension dataset with its own terms.
