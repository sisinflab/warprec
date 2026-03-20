import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FOLDER = os.path.join('analysis')
filename = "user_alpha_beta_precision.csv"
records_path = os.path.join(OUTPUT_FOLDER, filename)

df = pd.read_csv(records_path, header=0)
if df.shape[1] == 1:
    # fallback if file is tab-separated
    df = pd.read_csv(records_path, sep="\t", header=0)

# ---- Analisi cross-alpha/beta per utente ----
df.to_csv(os.path.join(OUTPUT_FOLDER, "user_alpha_beta_precision.csv"), index=False, sep='\t')

# miglior combinazione per utente
best_idx = df.groupby("user_id")["precision@10"].idxmax()
best = df.loc[best_idx].reset_index(drop=True)

# sensibilità: quanto varia la performance rispetto alla media utente
user_stats = (
    df.groupby("user_id")["precision@10"]
    .agg(mean="mean", std="std", max="max", min="min")
    .reset_index()
)
user_stats["delta_max_mean"] = user_stats["max"] - user_stats["mean"]

# distribuzioni globali dei migliori alpha/beta
best_alpha_dist = best["alpha"].value_counts().sort_index()
best_beta_dist = best["beta"].value_counts().sort_index()

print("\nTop 10 utenti più sensibili (delta_max_mean):")
print(user_stats.sort_values("delta_max_mean", ascending=False).head(10))

print("\nDistribuzione dei migliori alpha:")
print(best_alpha_dist)
print("\nDistribuzione dei migliori beta:")
print(best_beta_dist)

# salva output per analisi esterna
out_dir = "analysis/plots"    
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "user_alpha_beta_precision.csv"), index=False)
best.to_csv(os.path.join(out_dir, "best_alpha_beta_per_user.csv"), index=False)
user_stats.to_csv(os.path.join(out_dir, "user_sensitivity_stats.csv"), index=False)
print(f"\nSalvati: {out_dir}/user_alpha_beta_precision.csv, {out_dir}/best_alpha_beta_per_user.csv, {out_dir}/user_sensitivity_stats.csv")

# ---- Plot: densità best alpha/beta (per sovrapposizione punti) ----
plt.figure(figsize=(6, 5))
plt.hexbin(best["alpha"], best["beta"], gridsize=18, cmap="viridis", mincnt=1)
plt.xlabel("Best alpha")
plt.ylabel("Best beta")
plt.title("Best (alpha, beta) per user - density")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.colorbar(label="Users count")
density_path = os.path.join(out_dir, "best_alpha_beta_density.png")
plt.tight_layout()
plt.savefig(density_path, dpi=160)
plt.close()
print(f"Saved plot: {density_path}")

# ---- Plot: scatter con jitter per ridurre overlap ----
def _jitter(vals, scale=0.015):
    return vals + (np.random.randn(len(vals)) * scale)

plt.figure(figsize=(6, 5))
plt.scatter(_jitter(best["alpha"]), _jitter(best["beta"]), s=10, alpha=0.35)
plt.xlabel("Best alpha (jittered)")
plt.ylabel("Best beta (jittered)")
plt.title("Best (alpha, beta) per user - jittered")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.grid(True, linewidth=0.3, alpha=0.4)
jitter_path = os.path.join(out_dir, "best_alpha_beta_scatter_jitter.png")
plt.tight_layout()
plt.savefig(jitter_path, dpi=160)
plt.close()
print(f"Saved plot: {jitter_path}")

# ---- Clustering utenti sulla curva performance ----
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # matrice user x (alpha,beta)
    alphas = sorted(df["alpha"].unique())
    betas = sorted(df["beta"].unique())
    pivot = df.pivot_table(
        index="user_id",
        columns=["alpha", "beta"],
        values="precision@10",
        aggfunc="mean",
        fill_value=0.0,
    )

    n_users = pivot.shape[0]
    n_clusters = min(4, n_users) if n_users >= 2 else 1
    if n_clusters >= 2:
        X = StandardScaler().fit_transform(pivot.values)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        # PCA per visualizzare i cluster
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        plt.figure(figsize=(6, 5))
        plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10, alpha=0.7)
        plt.title("User clusters (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        pca_path = os.path.join(out_dir, "user_clusters_pca.png")
        plt.tight_layout()
        plt.savefig(pca_path, dpi=160)
        plt.close()
        print(f"Saved plot: {pca_path}")

        # heatmap media per cluster
        for c in range(n_clusters):
            users_in_c = pivot.index[labels == c]
            df_c = df[df["user_id"].isin(users_in_c)]
            mat = df_c.pivot_table(
                index="alpha",
                columns="beta",
                values="precision@10",
                aggfunc="mean",
            ).reindex(index=alphas, columns=betas)

            plt.figure(figsize=(6, 5))
            plt.imshow(mat.values, origin="lower", aspect="auto")
            plt.xticks(range(len(betas)), [f"{b:.1f}" for b in betas], rotation=45)
            plt.yticks(range(len(alphas)), [f"{a:.1f}" for a in alphas])
            plt.colorbar(label="Precision@10")
            plt.title(f"Cluster {c} mean heatmap")
            plt.xlabel("beta")
            plt.ylabel("alpha")
            heat_path = os.path.join(out_dir, f"cluster_{c}_heatmap.png")
            plt.tight_layout()
            plt.savefig(heat_path, dpi=160)
            plt.close()
            print(f"Saved plot: {heat_path}")
    else:
        print("Not enough users for clustering.")
except Exception as exc:  # noqa: BLE001
    print(f"Clustering/plot skipped: {exc}")