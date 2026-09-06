# RL-AC · RL-based adaptive access control against vehicular misbehavior

Reinforcement-learning **pseudonym-level adaptive access control** that defends
vehicle-to-everything (V2X/ITS) message exchanges against **DoS / Sybil / mixed
misbehavior**, evaluated against an **adaptive finite-state attacker** on the
VeReMi-Extension dataset.

> Status: refactored research codebase (previously an ad-hoc `code/` package,
> now the semantic `rlac` package). Paper route = RL-simulation only.

---

## Project framework

```
 VeReMi Extension  misbehavior traces (BSM)
        │  test-go/main.go  (Go parser)
        ▼
 flat message CSV  (time, sender_pseudo, receiver_id, pos_x/y, attack_type)
        │  rlac.data.datasets  (window slicing + 18 features)
        ▼
 per-window graph snapshot  (nodes = sender pseudonyms,
        │                     edges = shared-receiver Jaccard similarity)
        │  rlac.run.train_gnn  (supervised malicious-pseudonym classifier)
        ▼
 frozen GNN risk encoder  ──►  rlac.env.risk (node + graph embedding)
        │                            │
        │          ┌─────────────────┴──────────────────┐
        │          ▼                                    ▼
        │   adaptive attacker (FSM:                   defender policies
        │   latent→approach→burst→cooldown)            rlac.agents
        │          │                                   {drqn, cb, threshold}
        │          │  rejection-rate feedback            │
        └──────────▼────────────────────────────────────▼
        rlac.env.env : per-window step loop
        │   state = (node_map, (sight×embed) per active pseudonym)
        │   action = access level {allow, degrade, deny} per pseudonym
        │   reward = utility − α·scale-aware leakage penalty  (3-window Pareto)
        ▼
 per-run artifacts: outputs/simulation/<save_name>/
        { npy/csv/png curves · final_agent.pt · rejection/pressure plots · run_meta.json }
```

### Responsibilities (why it is split this way)

| Module | Responsibility | Replaces |
|---|---|---|
| `rlac/config.py` | typed/validated run config (YAML → dataclasses) | ad-hoc `yaml.safe_load` + `cfg.get` |
| `rlac/data/` | feature registry, graph-snapshot builder, dataset builders | `code/data/*` |
| `rlac/env/` | windowed replay environment + managers (data/attacker/graph/node/policy/risk) | `code/env_entity/*` |
| `rlac/agents/` | defender policies & replay buffers (DRQN / CB / threshold) | `code/entity_agent/*` + inline `GNNThresholdAgent` |
| `rlac/models/` | GNN / MLP / STGCN encoders | `code/model/*` |
| `rlac/run/` | thin entry points: `simulate`, `train_gnn` | `code/train/simulation.py`, `code/train/gnn_train.py` |

---

## Repository layout

```
data_view/
├── rlac/                  # ★ python package (production pipeline)
│   ├── config.py          #   typed config schema + loader
│   ├── data/              #   features.py · builder.py · datasets.py
│   ├── env/               #   data_module · attacker · graph_module ·
│   │                      #   node_manager · policy · risk · env
│   ├── agents/            #   replay · drqn · cb · threshold
│   ├── models/            #   gnn · mlp · stgcn
│   ├── run/               #   simulate.py (train/test CLI) · train_gnn.py
│   └── utils/random.py    #   seed_all (python+numpy+torch)
├── config/                # YAML run configs (incl. config/_smoke_*.yaml)
├── figure/                # paper figures (pdf+svg+png)
├── data/                  # feature_names.txt (raw CSVs live at repo root)
├── test-go/               # VeReMi trace parser (Go)
├── main.sh                # experiment-matrix runner (rlac entrypoints)
├── build_*.sh             # per-attack ingest + dataset + GNN encoder
├── outputs/               # run artifacts / scalers / models (git-ignored)
├── logs/  legacy_removed  # run-history provenance
├── environment.yml · requirements.txt · pyproject.toml · LICENSE · README.md
└── tests/                 # pytest (config load + RNG determinism)
```

Run artifacts and raw data (`outputs/`, `*.csv`, `*.pt`, `*.npz`) are
git-ignored and regenerated locally.

## Environment

```bash
conda env create -f environment.yml    # or: pip install -r requirements.txt
```
> On Windows set `PYTHONIOENCODING=utf-8` (progress prints use emoji that GBK
> consoles cannot encode).

## Reproduce (from raw data)

```bash
# 1) parse a VeReMi scenario into messages CSV
go run test-go/main.go <VeReMi-dir> <start> <end>
cp output.csv sybil.csv                # or dos.csv / dosrandom.csv …

# 2) build graph snapshots + train a supervised GNN risk encoder
python -m rlac.data.datasets --data_path output.csv --attack_intensity_list 1.0
python -m rlac.run.train_gnn --exp_name single_sybil_gnn --use_timestamp 0

# 3) train / test a defender
python -m rlac.run.simulate --config config/train_rl.yaml

# or the full experiment matrix
bash main.sh
```

### Minimal smoke
```bash
export PYTHONIOENCODING=utf-8
python -m rlac.run.simulate --config config/_smoke_threshold.yaml   # threshold test
python -m rlac.run.simulate --config config/_smoke_rl.yaml          # RL train
python -m rlac.run.simulate --config config/_smoke_cb.yaml          # CB train
```

## Outputs & naming
`outputs/simulation/<save_name>/{png,csv,npy}/`, `final_agent.pt`, plus
`run_meta.json` (config + seed + git rev) for provenance. `save_name` encodes
scenario · method · alpha · sight · train/test seed (see `main.sh`).

## Fixed correctness / reproducibility issues
- Discounting consistent: replay stores **2-step** returns; DQN bootstraps `γ²`.
- Contextual-bandit baseline regresses on **single-step immediate rewards** (no look-ahead).
- Attacker rejection-rate is derived from **admitted weight `1−w`** (works for
  2- and 3-level actions) and its RNG is **seeded** from the run seed.
- All RNGs (python `random`, numpy, torch, data subsampling) are seeded.

## License
MIT — see `LICENSE`. Data: VeReMi-Extension (public dataset, its own terms).
