# AGENTS.md

## What is this repo

CIKGRec (AAAI-2025) — a knowledge-graph recommender system that uses LLMs to infer user-side interests. PyTorch + torch-geometric.

## Run

```bash
python -u train.py --dataset book-crossing   # or: ml1m, dbbook2014
```

Config is read from `config/<dataset>.json`. Default dataset is `dbbook2014`.

## Dependencies

```bash
pip install torch==1.13.1 torch-geometric==2.3.1 scikit-learn==1.0.2 sentence-transformers==3.0.0 prettytable==3.10.0 nltk==3.8.1 jsonlines
```

No `requirements.txt` — install manually. CUDA required (device set in config JSON).

## Data pipeline (3 steps, run in order)

1. `User_Interest_Generation_BatchMode.py` — generates JSONL input for OpenAI batch API. Writes to `batch_input/`.
2. Upload the JSONL to OpenAI batch endpoint, download output to `batch_output/`.
3. `Structration_User_Knowledge.py` — clusters LLM output into `user_interest_clustered.txt`. Requires sentence-transformers or TF-IDF.

Pre-processed data already exists in `data/<dataset>/`. Step 1-3 only needed to regenerate user interest graphs.

## Key files

| File | Role |
|---|---|
| `train.py` | Entry point — builds graphs, trains model, evaluates |
| `models/Model.py` | `CIKGRec` class — GNN forward, contrastive loss, interest reconstruction |
| `models/loss_func.py` | BPR, InfoNCE, SCE loss functions |
| `utils.py` | Data loading (`MyLoader`), graph utilities, `TrainDataset` |
| `evaluate.py` | Evaluation metrics (recall, NDCG, precision, hit ratio) |
| `call_llm.py` | Prompt templates for LLM inference |
| `config/*.json` | Per-dataset hyperparameters |

## Architecture notes

- `MyLoader` remaps item IDs by adding `user_num` offset — item IDs start after user IDs in the unified embedding space.
- `user_interest_clustered.txt` interest IDs must be remapped to start after the entity range (handled in `MyLoader.loadData`).
- Evaluation uses multiprocessing (`cpu_count() // 2` cores). Can hang if pool is not properly terminated on error.
- `use_kge` controls whether TransE training runs after CF training (enabled only for `dbbook2014`).
- Dynamic mask rate for interest reconstruction: increases from `min_mask_rate` to `max_mask_rate` over `total_epoch` steps (exponential by default).

## Gotchas

- No validation set split in the data — `get_eval_data('valid')` returns the same train rated_dict, test evaluation uses `item_dict_test`.
- The `eval_interval` config controls how often evaluation runs (default 1 = every epoch).
- Early stopping uses `patience` (default 10 epochs).
- Best model selection: sum of `recall@1` + `ndcg@1` where index 1 corresponds to `Ks[1]` (50 by default).
