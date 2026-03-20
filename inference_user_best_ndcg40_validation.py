#!/usr/bin/env python3
"""
Calcola il best alpha/beta per utente su validation, usando nDCG@40.
Salva un CSV con la migliore combinazione per ogni utente e stampa la media nDCG@40.
"""

import os
import pandas as pd

from inference_validation_grid import (
    MODELS_CONFIG,
    DATASET_PATH,
    OUTPUT_BASE_DIR,
    ALPHA_VALUES,
    BETA_VALUES,
    LIMIT_REC,
    load_legacy_split_data,
    load_model_from_checkpoint,
    create_sessions_from_dataframe,
    generate_recommendations_generic,
)


def user_ndcg_at_k(user_recs, user_relevant_items, k=40):
    """nDCG@k per singolo utente."""
    top_k_items = [str(item_id) for item_id, _ in user_recs[:k]]
    if not user_relevant_items:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(top_k_items):
        if item_id in user_relevant_items:
            dcg += 1.0 / (i + 2)

    idcg = sum(1.0 / (i + 2) for i in range(min(len(user_relevant_items), k)))
    return dcg / idcg if idcg > 0 else 0.0


def user_metrics_at_k(user_recs, user_relevant_items, k=40):
    """Precision@k, Recall@k, nDCG@k per singolo utente."""
    top_k_items = [str(item_id) for item_id, _ in user_recs[:k]]
    if not user_relevant_items:
        return 0.0, 0.0, 0.0

    hits = len(set(top_k_items) & user_relevant_items)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(user_relevant_items) if len(user_relevant_items) > 0 else 0.0
    ndcg = user_ndcg_at_k(user_recs, user_relevant_items, k=k)
    return precision, recall, ndcg


def build_gt_items_per_user(gt_df):
    gt_items_per_user = {}
    for user_id in gt_df["user_id"].unique():
        user_items = gt_df.filter(gt_df["user_id"] == user_id)["item_id"].to_list()
        gt_items_per_user[int(user_id)] = set(user_items)
    return gt_items_per_user


def main():
    print("=" * 80)
    print("BEST ALPHA/BETA PER UTENTE SU VALIDATION (nDCG@40)")
    print("=" * 80)

    train_df, _test_df, val_df = load_legacy_split_data(DATASET_PATH)
    if val_df is None:
        raise ValueError("Val set non trovato. Serve validation.csv")

    val_users = sorted(val_df["user_id"].unique().to_list())
    gt_items_per_user = build_gt_items_per_user(val_df)
    sessions = create_sessions_from_dataframe(train_df)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    for model_name, config in MODELS_CONFIG.items():
        if not config["enabled"]:
            continue

        print(f"\n{'=' * 80}")
        print(f"📊 PROCESSING: {model_name}")
        print(f"{'=' * 80}")

        checkpoint_path = config["checkpoint"]
        model, device, info, _params = load_model_from_checkpoint(checkpoint_path, model_name)

        supports_alpha_beta = (
            hasattr(model, "item_codes_layer")
            and hasattr(model.item_codes_layer, "alpha")
            and hasattr(model.item_codes_layer, "beta")
        )
        if not supports_alpha_beta:
            print(f"❌ {model_name} non supporta alpha/beta")
            continue

        best_per_user = {}

        # Grid alpha/beta su validation
        for alpha in ALPHA_VALUES:
            for beta in BETA_VALUES:
                if alpha + beta > 1.0:
                    continue

                gamma = 1.0 - alpha - beta
                if gamma < 0.0:
                    continue

                model.item_codes_layer.alpha.data.fill_(alpha)
                model.item_codes_layer.beta.data.fill_(beta)

                recommendations = generate_recommendations_generic(
                    model=model,
                    sessions=sessions,
                    eval_users=val_users,
                    device=device,
                    limit=LIMIT_REC,
                    info=info,
                )

                # Calcolo nDCG@40 per utente e update best (stile idxmax: in pari tiene prima)
                for user_id in val_users:
                    if user_id not in recommendations or user_id not in gt_items_per_user:
                        continue

                    ndcg40 = user_ndcg_at_k(
                        user_recs=recommendations[user_id],
                        user_relevant_items=gt_items_per_user[user_id],
                        k=40,
                    )

                    p10, r10, n10 = user_metrics_at_k(
                        user_recs=recommendations[user_id],
                        user_relevant_items=gt_items_per_user[user_id],
                        k=10,
                    )
                    p40, r40, n40 = user_metrics_at_k(
                        user_recs=recommendations[user_id],
                        user_relevant_items=gt_items_per_user[user_id],
                        k=40,
                    )
                    p100, r100, n100 = user_metrics_at_k(
                        user_recs=recommendations[user_id],
                        user_relevant_items=gt_items_per_user[user_id],
                        k=100,
                    )

                    prev = best_per_user.get(user_id)
                    if prev is None or ndcg40 > prev["nDCG@40"]:
                        best_per_user[user_id] = {
                            "model": model_name,
                            "user_id": user_id,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "Precision@10": p10,
                            "Recall@10": r10,
                            "nDCG@10": n10,
                            "Precision@40": p40,
                            "Recall@40": r40,
                            "nDCG@40": n40,
                            "Precision@100": p100,
                            "Recall@100": r100,
                            "nDCG@100": n100,
                        }

        if not best_per_user:
            print("⚠️ Nessun best per-user calcolato")
            continue

        # Salva best per utente
        df_best = pd.DataFrame(list(best_per_user.values())).sort_values("user_id")
        out_csv = os.path.join(OUTPUT_BASE_DIR, f"{model_name}_best_per_user_ndcg40_validation.csv")
        df_best.to_csv(out_csv, index=False)

        # Medie metriche sui best per-user (selezionati su base nDCG@40)
        metric_cols = [
            "Precision@10", "Recall@10", "nDCG@10",
            "Precision@40", "Recall@40", "nDCG@40",
            "Precision@100", "Recall@100", "nDCG@100",
        ]
        means = {m: float(df_best[m].mean()) for m in metric_cols if m in df_best.columns}

        # Salva anche summary
        summary_csv = os.path.join(OUTPUT_BASE_DIR, f"{model_name}_summary_best_per_user_ndcg40_validation.csv")
        summary_row = {
            "model": model_name,
            "num_users": len(df_best),
        }
        summary_row.update({f"mean_{k}": v for k, v in means.items()})
        pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

        print(f"✅ Salvato: {out_csv}")
        print(f"✅ Salvato: {summary_csv}")
        print("📈 Medie metriche sui best per-user (criterio: nDCG@40):")
        for m in metric_cols:
            if m in means:
                print(f"   {m:15s}: {means[m]:.6f}")


if __name__ == "__main__":
    main()
