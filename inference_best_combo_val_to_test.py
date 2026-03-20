#!/usr/bin/env python3
"""
Seleziona la seconda migliore combinazione alpha/beta/gamma su validation
(ordinando per nDCG@40) e poi la valuta su test.csv.
"""

import os
import pandas as pd

from inference_validation_grid import (
    MODELS_CONFIG,
    DATASET_PATH,
    OUTPUT_BASE_DIR,
    load_legacy_split_data,
    load_model_from_checkpoint,
    create_sessions_from_dataframe,
    generate_recommendations_generic,
    compute_metrics,
)


def main():
    print("=" * 80)
    print("SECOND BEST COMBO SU VALIDATION -> TEST FINALE SU TEST.CSV")
    print("=" * 80)

    grid_csv = os.path.join(OUTPUT_BASE_DIR, "all_alpha_beta_results_validation.csv")
    if not os.path.exists(grid_csv):
        raise FileNotFoundError(
            f"File grid non trovato: {grid_csv}. Esegui prima inference_validation_grid.py"
        )
    grid_df = pd.read_csv(grid_csv)

    train_df, test_df, _val_df = load_legacy_split_data(DATASET_PATH)
    test_users = sorted(test_df["user_id"].unique().to_list())

    sessions = create_sessions_from_dataframe(train_df)
    final_rows = []

    for model_name, config in MODELS_CONFIG.items():
        if not config["enabled"]:
            continue

        print(f"\n{'=' * 80}\n📊 PROCESSING: {model_name}\n{'=' * 80}")
        checkpoint_path = config["checkpoint"]

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

            model_grid = grid_df[grid_df["model"] == model_name].copy()
            if model_grid.empty:
                raise RuntimeError(f"Nessuna riga grid trovata nel CSV per modello {model_name}")
            if "nDCG@40" not in model_grid.columns:
                raise RuntimeError("Colonna nDCG@40 mancante nel CSV grid")

            model_grid_sorted = model_grid.sort_values("nDCG@40", ascending=False).reset_index(drop=True)
            if len(model_grid_sorted) < 2:
                raise RuntimeError(
                    f"Combinazioni insufficienti per modello {model_name}: servono almeno 2 righe"
                )

            # Seconda migliore combinazione per nDCG@40
            best_row = model_grid_sorted.iloc[1]
            best_combo = {
                "alpha": float(best_row["alpha"]),
                "beta": float(best_row["beta"]),
                "gamma": float(best_row["gamma"]),
                "val_nDCG@40": float(best_row["nDCG@40"]),
            }

            # Test finale con best combo da validation
            model.item_codes_layer.alpha.data.fill_(best_combo["alpha"])
            model.item_codes_layer.beta.data.fill_(best_combo["beta"])

            test_recs = generate_recommendations_generic(
                model=model,
                sessions=sessions,
                eval_users=test_users,
                device=device,
                limit=100,
                info=info,
            )
            test_metrics = compute_metrics(test_recs, test_df, k_values=[10, 40, 100])

            row = {
                "model": model_name,
                "alpha": best_combo["alpha"],
                "beta": best_combo["beta"],
                "gamma": best_combo["gamma"],
                "val_nDCG@40": best_combo["val_nDCG@40"],
            }
            row.update(test_metrics)
            final_rows.append(row)

            print("Second best combo su validation:")
            print(
                f"alpha={best_combo['alpha']:.2f}, beta={best_combo['beta']:.2f}, "
                f"gamma={best_combo['gamma']:.2f}, nDCG@40={best_combo['val_nDCG@40']:.6f}"
            )
            print("Metriche su test.csv:")
            for k, v in test_metrics.items():
                print(f"  {k:20s}: {v:.6f}")

        except Exception as exc:
            print(f"❌ ERRORE per {model_name}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    if final_rows:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        out_csv = os.path.join(OUTPUT_BASE_DIR, "best_combo_from_val_test_metrics.csv")
        metric_cols = [
            "Precision@10", "Recall@10", "nDCG@10",
            "Precision@40", "Recall@40", "nDCG@40",
            "Precision@100", "Recall@100", "nDCG@100",
        ]
        cols = ["model", "alpha", "beta", "gamma", "val_nDCG@40"] + metric_cols
        df = pd.DataFrame(final_rows)
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(out_csv, index=False)
        print(f"\n✅ Salvato: {out_csv}")


if __name__ == "__main__":
    main()
