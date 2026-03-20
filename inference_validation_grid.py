#!/usr/bin/env python3
"""
Grid search alpha/beta su VALIDATION set (non test).
Salva raccomandazioni e metriche per ogni combinazione.
"""

import os
import csv
import torch
import pandas as pd
import polars as pl

from warprec.utils.registry import model_registry

try:
    import warprec_recjpq  # noqa: F401
except ImportError:
    print("⚠️  Modulo warprec_recjpq non trovato - alcuni modelli potrebbero non essere disponibili")


# ============================================================================
# CONFIG
# ============================================================================
MODELS_CONFIG = {
    # "SASRecBase": {
    #     "checkpoint": "/home/chiara/projects/warprec/experiments/lastfm1k/legacy_split_raw_jpq/lastfm1k_30000_legacy_split_raw_jpq/ray_results/objective_function_2026-03-07_06-06-08/SASRec_gts_3465f893_1_batch_size=128,dropout_prob=0.2000,embedding_size=128,epochs=10000,learning_rate=0.0100,max_seq_len=150,n_he_2026-03-07_06-06-08/checkpoint_004884/checkpoint.pt",
    #     "enabled": True,
    # },
    "SASRecJPQMix": {
        "checkpoint": "/home/chiara/projects/warprec/experiments/lastfm1k/legacy_split_raw_jpq/lastfm1k_30000_legacy_split_raw_jpq/ray_results/objective_function_2026-03-11_16-24-25/SASRecJPQMix_95101655_0_batch_size=128,centroid_strategy=svd,dropout_prob=0.2000,embedding_size=128,epochs=10000,inner_size=128,le_2026-03-11_16-24-25/checkpoint_002196/checkpoint.pt",
        "enabled": True,
    },
}

DATASET_PATH = "data/lastfm1k_30000/legacy_global_split_raw"
OUTPUT_BASE_DIR = "./inference_output_validation"
PATH_TO_ORIGINAL_TSV = "/home/chiara/projects/warprec/data/lastfm1k_30000/userid-timestamp-artid-artname-traid-traname.tsv"
LIMIT_REC = 100

ALPHA_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BETA_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MODEL_NAME_ALIASES = {
    "SASRecBase": "SASREC_GTS",
    "SASRec": "SASREC",
    "gSASRec": "GSASREC",
    "BertJPQMix": "BERTJPQMIX",
    "SASRecJPQMix": "SASRECJPQMIX",
    "gSASRecJPQMix": "GSASRECJPQMIX",
}


# ============================================================================
# HELPERS
# ============================================================================
def resolve_model_registry_name(requested_name: str) -> str:
    available = list(model_registry._registry.keys()) if hasattr(model_registry, "_registry") else []

    aliased = MODEL_NAME_ALIASES.get(requested_name, requested_name)
    if aliased in available:
        return aliased

    upper_map = {name.upper(): name for name in available}
    if aliased.upper() in upper_map:
        return upper_map[aliased.upper()]

    def norm(s: str) -> str:
        return s.replace("_", "").replace("-", "").upper()

    target = norm(aliased)
    for name in available:
        if norm(name) == target:
            return name

    raise ValueError(f"Modello '{requested_name}' non trovato nel registry. Disponibili: {available}")


def load_lastfm_metadata(tsv_path: str):
    print(f"Caricamento metadati da {tsv_path}...")
    mapping = {}
    if not os.path.exists(tsv_path):
        print(f"ERRORE: File {tsv_path} non trovato!")
        return mapping

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                track_id = parts[4]
                artist_name = parts[3]
                track_name = parts[5]
                if track_id and track_id not in mapping:
                    mapping[track_id] = (artist_name, track_name)

    print(f"Mappatura completata: {len(mapping)} brani caricati.")
    return mapping


def load_legacy_split_data(data_path: str):
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")
    val_path = os.path.join(data_path, "validation.csv")

    print(f"Caricamento split da: {data_path}")

    schema = {
        "user_id": pl.Int64,
        "item_id": pl.Utf8,
        "rating": pl.Float64,
        "timestamp": pl.Utf8,
    }

    train_df = pl.read_csv(train_path, has_header=False, new_columns=list(schema.keys()), schema=schema)
    test_df = pl.read_csv(test_path, has_header=False, new_columns=list(schema.keys()), schema=schema)
    val_df = pl.read_csv(val_path, has_header=False, new_columns=list(schema.keys()), schema=schema) if os.path.exists(val_path) else None

    print(f"Train: {len(train_df)} interazioni, {train_df['user_id'].n_unique()} utenti")
    print(f"Test: {len(test_df)} interazioni, {test_df['user_id'].n_unique()} utenti")
    if val_df is not None:
        print(f"Val: {len(val_df)} interazioni, {val_df['user_id'].n_unique()} utenti")

    return train_df, test_df, val_df


def load_model_from_checkpoint(checkpoint_path: str, model_name: str):
    print(f"\nCaricamento modello '{model_name}' da: {checkpoint_path}")

    if checkpoint_path.endswith(".pt") and os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
    else:
        checkpoint_file = os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    info = checkpoint.get("info", {})
    params = checkpoint.get("params", {})
    if not params and "hyperparameters" in checkpoint:
        params = checkpoint["hyperparameters"]

    resolved_name = resolve_model_registry_name(model_name)
    if resolved_name != model_name:
        print(f"Alias modello: '{model_name}' -> '{resolved_name}'")

    model_class = model_registry.get_class(resolved_name)
    model = model_class(params=params, info=info, seed=42)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device, info, params


def create_sessions_from_dataframe(train_df: pl.DataFrame):
    class MockSessions:
        def __init__(self, df):
            self._df = df
            self.user_label = "user_id"
            self.item_label = "item_id"
            self.timestamp_label = "timestamp"
            self.rating_label = "rating"

            users = df["user_id"].unique().sort()
            self._user_offsets = [0]
            self._flat_items = []

            for user_id in users:
                user_data = df.filter(pl.col("user_id") == user_id).sort("timestamp")
                items = user_data["item_id"].to_list()
                self._flat_items.extend(items)
                self._user_offsets.append(len(self._flat_items))

        def _get_processed_data(self):
            return self._df

    return MockSessions(train_df)


def generate_recommendations_generic(model, sessions, eval_users, device, limit: int = 100, info: dict = None):
    print(f"\nGenerazione raccomandazioni Top-{limit} per {len(eval_users)} utenti...")

    item_mapping = info.get("item_mapping", {}) if info else {}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()} if item_mapping else {}

    recommendations = {}

    # fallback robusto (manuale)
    df = sessions._get_processed_data()
    with torch.no_grad():
        for user_id in eval_users:
            user_data = df.filter(pl.col("user_id") == user_id).sort("timestamp")
            if len(user_data) == 0:
                recommendations[user_id] = [(0, 0.0)] * limit
                continue

            items = user_data["item_id"].to_list()
            items_indices = [item_mapping.get(item, 0) for item in items] if item_mapping and items and isinstance(items[0], str) else items

            if not hasattr(model, "max_seq_len"):
                recommendations[user_id] = [(0, 0.0)] * limit
                continue

            max_len = model.max_seq_len
            if len(items_indices) > max_len:
                items_indices = items_indices[-max_len:]

            if hasattr(model, "mask_token_id"):
                pad_id = getattr(model, "padding_token_id", 0)
                padded = [pad_id] * (max_len - len(items_indices) - 1) + items_indices + [model.mask_token_id]
            else:
                pad_id = getattr(model, "padding_token_id", 0)
                padded = [pad_id] * (max_len - len(items_indices)) + items_indices

            seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            seq_len = torch.tensor([len(items_indices)], dtype=torch.long, device=device)
            user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)

            try:
                if hasattr(model, "predict"):
                    scores = model.predict(user_indices=user_tensor, user_seq=seq_tensor, seq_len=seq_len)
                else:
                    scores = model.forward(seq_tensor)
                    if scores.dim() == 3:
                        scores = scores[:, -1, :]
                    if hasattr(model, "item_codes_layer"):
                        scores = model.item_codes_layer.score_all_items(scores)

                top_k_scores, top_k_items = torch.topk(scores[0], limit)

                if reverse_item_mapping:
                    recs = [
                        (reverse_item_mapping.get(int(top_k_items[i].item()), str(top_k_items[i].item())), float(top_k_scores[i].item()))
                        for i in range(limit)
                    ]
                else:
                    recs = [(int(top_k_items[i].item()), float(top_k_scores[i].item())) for i in range(limit)]

                recommendations[user_id] = recs
            except Exception as e:
                print(f"Errore per utente {user_id}: {e}")
                recommendations[user_id] = [(0, 0.0)] * limit

    return recommendations


def save_recommendations_csv(recommendations: dict, item_metadata: dict, output_path: str):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "rank", "item_id", "artist_name", "track_name", "score"])

        for user_id in sorted(recommendations.keys()):
            for rank, (item_id, score) in enumerate(recommendations[user_id], start=1):
                item_id_str = str(item_id)
                artist, track = item_metadata.get(item_id_str, (item_id_str, "Unknown Track"))
                writer.writerow([user_id, rank, item_id_str, artist, track, score])


def compute_metrics(recommendations: dict, gt_df: pl.DataFrame, k_values=[10, 40, 100]):
    gt_items_per_user = {}
    for user_id in gt_df["user_id"].unique():
        user_items = gt_df.filter(pl.col("user_id") == user_id)["item_id"].to_list()
        gt_items_per_user[int(user_id)] = set(user_items)

    results = {}
    for k in k_values:
        precisions, recalls, ndcgs = [], [], []

        for user_id, recs in recommendations.items():
            if user_id not in gt_items_per_user:
                continue

            top_k_items = [str(item_id) for item_id, _ in recs[:k]]
            relevant_items = gt_items_per_user[user_id]

            hits = len(set(top_k_items) & relevant_items)
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

            dcg = 0.0
            for i, item_id in enumerate(top_k_items):
                if item_id in relevant_items:
                    dcg += 1.0 / (i + 2)
            idcg = sum(1.0 / (i + 2) for i in range(min(len(relevant_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        results[f"Precision@{k}"] = sum(precisions) / len(precisions) if precisions else 0.0
        results[f"Recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        results[f"nDCG@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    return results


def build_gt_items_per_user(gt_df: pl.DataFrame):
    gt_items_per_user = {}
    for user_id in gt_df["user_id"].unique():
        user_items = gt_df.filter(pl.col("user_id") == user_id)["item_id"].to_list()
        gt_items_per_user[int(user_id)] = set(user_items)
    return gt_items_per_user


def compute_user_metrics(recommendations: dict, gt_items_per_user: dict, k_values=(10, 40, 100)):
    """Calcola metriche per utente (non aggregate)."""
    user_rows = {}
    for user_id, recs in recommendations.items():
        if user_id not in gt_items_per_user:
            continue

        relevant_items = gt_items_per_user[user_id]
        row = {"user_id": user_id}

        for k in k_values:
            top_k_items = [str(item_id) for item_id, _ in recs[:k]]
            hits = len(set(top_k_items) & relevant_items)
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

            dcg = 0.0
            for i, item_id in enumerate(top_k_items):
                if item_id in relevant_items:
                    dcg += 1.0 / (i + 2)
            idcg = sum(1.0 / (i + 2) for i in range(min(len(relevant_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            row[f"Precision@{k}"] = precision
            row[f"Recall@{k}"] = recall
            row[f"nDCG@{k}"] = ndcg

        user_rows[user_id] = row

    return user_rows


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("GRID SEARCH SU VALIDATION")
    print("=" * 80)

    train_df, _test_df, val_df = load_legacy_split_data(DATASET_PATH)

    if val_df is None:
        raise ValueError("Val set non trovato. Serve val.csv per fare grid su validation.")

    val_users = sorted(val_df["user_id"].unique().to_list())
    print(f"\nUtenti validation totali: {len(val_users)}")

    # Sessioni costruite SOLO sul train
    sessions = create_sessions_from_dataframe(train_df)
    all_results = []

    for model_name, config in MODELS_CONFIG.items():
        if not config["enabled"]:
            continue

        checkpoint_path = config["checkpoint"]
        print(f"\n{'=' * 80}\n📊 PROCESSING: {model_name}\n{'=' * 80}")

        try:
            model, device, info, _params = load_model_from_checkpoint(checkpoint_path, model_name)

            supports_alpha_beta = (
                hasattr(model, "item_codes_layer")
                and hasattr(model.item_codes_layer, "alpha")
                and hasattr(model.item_codes_layer, "beta")
            )

            if not supports_alpha_beta:
                print(f"❌ {model_name} non supporta alpha/beta")
                continue

            print(f"\n🔍 Grid Search alpha/beta su VALIDATION per {model_name}")

            for alpha in ALPHA_VALUES:
                for beta in BETA_VALUES:
                    if alpha + beta > 1.0:
                        continue

                    gamma = 1.0 - alpha - beta
                    if gamma < 0.0:
                        continue

                    model.item_codes_layer.alpha.data.fill_(alpha)
                    model.item_codes_layer.beta.data.fill_(beta)

                    val_recommendations = generate_recommendations_generic(
                        model=model,
                        sessions=sessions,
                        eval_users=val_users,
                        device=device,
                        limit=LIMIT_REC,
                        info=info,
                    )
                    val_metrics = compute_metrics(val_recommendations, val_df, k_values=[10, 40, 100])

                    result_row = {
                        "model": model_name,
                        "run_name": f"{model_name}_alpha{alpha:.2f}_beta{beta:.2f}_gamma{gamma:.2f}",
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                    }
                    result_row.update(val_metrics)
                    all_results.append(result_row)

            print(f"✅ Grid completata per {model_name}")

        except Exception as e:
            print(f"❌ ERRORE per {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        results_csv = os.path.join(OUTPUT_BASE_DIR, "all_alpha_beta_results_validation.csv")

        metric_cols = [
            "Precision@10", "Recall@10", "nDCG@10",
            "Precision@40", "Recall@40", "nDCG@40",
            "Precision@100", "Recall@100", "nDCG@100",
        ]
        base_cols = ["model", "run_name", "alpha", "beta", "gamma"]

        df_results = pd.DataFrame(all_results)
        ordered_cols = [c for c in (base_cols + metric_cols) if c in df_results.columns]
        df_results = df_results[ordered_cols]
        df_results.to_csv(results_csv, index=False)
        print(f"\n✅ Risultati salvati in: {results_csv}")

        if "nDCG@40" in df_results.columns:
            best_row = df_results.loc[df_results["nDCG@40"].idxmax()]
            print("\n🏆 Best combinazione su validation (nDCG@40):")
            print(
                f"alpha={best_row['alpha']:.2f}, beta={best_row['beta']:.2f}, "
                f"gamma={best_row['gamma']:.2f}, nDCG@40={best_row['nDCG@40']:.6f}"
            )


if __name__ == "__main__":
    main()
