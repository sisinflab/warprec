
from datarec.datasets import Gowalla
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import pandas as pd
from datarec.io.readers.transactions import read_transactions_tabular


dataset_path = 'data/lastfm-1k/userid-timestamp-artid-artname-traid-traname_fix.tsv'

dataset = "lastfm"  # Change to your dataset of choice

if dataset == "lastfm":
    if not os.path.exists(dataset_path):
        raise Exception(f"Expected fixed dataset file not found: {dataset_path}. Please run the preprocessing step to create it.")
    print(f"Loading dataset from: {dataset_path}")
    dr = read_transactions_tabular(dataset_path, 
                                   sep='\t',
                                   user_col=0,
                                   item_col=1,
                                   timestamp_col=2)
elif dataset == "gowalla":
    dr = Gowalla().prepare_and_load()
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

os.makedirs("figures", exist_ok=True)

repetitions = {}

repetitions_ratio = {}
repetitions_frequency= defaultdict(list)
repetitions_num_prob = defaultdict(list)
ert_deltas = defaultdict(list)
ert_norm_deltas = defaultdict(list)

points = 0

# check if there are any duplicate user-item interactions
for user in tqdm(dr.users, desc="Users"):
    udr = dr.data[dr.data['user_id'] == user]
    # sort by timestamp (fallback to row order if missing or invalid)
    time_col = None
    for col in ("timestamp", "time", "datetime", "created_at"):
        if col in udr.columns:
            time_col = col
            break
    if time_col is not None:
        udr = udr.sort_values(time_col)
        times = pd.to_datetime(udr[time_col], errors="coerce")
        if times.isna().any():
            times = pd.Series(np.arange(len(udr)), index=udr.index)
        else:
            times = times.astype("int64")
    else:
        times = pd.Series(np.arange(len(udr)), index=udr.index)

    unique_items = udr['item_id'].unique()
    n_unique_items = len(unique_items)
    item_counts = Counter(udr['item_id'])
    items_by_count = defaultdict(list)
    for item, count in item_counts.items():
        items_by_count[count].append(item)
    items_by_count = dict(items_by_count)
    # if points == 1000:
    #     break
    # points += 1

    # repetition ratio
    # aggiungere correzioen per caso limite n_unique_items == 1
    rr = 1 - (n_unique_items) / len(udr)
    repetitions_ratio[user] = rr

    # repetitions frequency
    for count, items in items_by_count.items():
        if count > 1:
            repetitions_frequency[count].append(len(items))

    # probability of repetition of length k
    for count in items_by_count:
        prob = len(items_by_count[count]) / n_unique_items
        repetitions_num_prob[count].append(prob)

    # expected return time (ERT) by occurrence index k
    item_times = defaultdict(list)
    for item, t in zip(udr['item_id'], times):
        item_times[item].append(t)
    profile_len = len(udr)
    for occs in item_times.values():
        if len(occs) < 2:
            continue
        for idx in range(len(occs) - 1):
            delta = occs[idx + 1] - occs[idx]
            k = idx + 1  # k-th to (k+1)-th occurrence
            ert_deltas[k].append(delta)
            if profile_len > 0:
                ert_norm_deltas[k].append(delta / profile_len)
    

# Plot distribution and mean
ratios = np.array(list(repetitions_ratio.values()), dtype=float)
mean_ratio = float(ratios.mean()) if len(ratios) else 0.0

plt.figure(figsize=(8, 5))
plt.hist(ratios, bins=40, color="#4C78A8", edgecolor="white", alpha=0.9)
plt.axvline(mean_ratio, color="#F58518", linestyle="--", linewidth=2, label=f"Mean: {mean_ratio:.4f}")
plt.title("User Repetition Ratio Distribution")
plt.xlabel("Repetition Ratio")
plt.ylabel("Users")
plt.legend()
plt.tight_layout()
plt.savefig("figures/repetition_ratio_distribution.png", dpi=150)
plt.show()

# Mean probability of repetition by length k
ks = sorted(repetitions_num_prob.keys())
mean_probs = [float(np.mean(repetitions_num_prob[k])) for k in ks]

plt.figure(figsize=(8, 5))
plt.plot(ks, mean_probs, marker="o", color="#54A24B")
plt.xscale("log")
plt.title("Mean Probability of Repetition by Length k")
plt.xlabel("Repetition Length k")
plt.ylabel("Mean Probability")
plt.tight_layout()
plt.savefig("figures/mean_prob_by_k.png", dpi=150)
plt.show()
    
# Histogram of repeated items by length k (sum over users)
ks_freq = sorted(repetitions_frequency.keys())
total_items = [int(sum(repetitions_frequency[k])) for k in ks_freq]

plt.figure(figsize=(8, 5))
plt.hist(ks_freq, bins=30, weights=total_items, color="#E45756", edgecolor="white")
plt.yscale("log")
plt.title("Histogram of Repeated Items by Length k")
plt.xlabel("Repetition Length k")
plt.ylabel("Total Repeated Items")
plt.tight_layout()
plt.savefig("figures/repeated_items_hist_k.png", dpi=150)
plt.show()

# Expected Return Time (ERT) summary DataFrame
ert_rows = []
for k in sorted(ert_deltas.keys()):
    deltas = np.array(ert_deltas[k], dtype=float)
    ert = float(deltas.mean()) if len(deltas) else np.nan
    norm_deltas = np.array(ert_norm_deltas[k], dtype=float)
    ert_norm = float(norm_deltas.mean()) if len(norm_deltas) else np.nan
    ert_rows.append({"k": k, "ert": ert, "count": len(deltas), "ert_norm": ert_norm})

ert_df = pd.DataFrame(ert_rows, columns=["k", "ert", "count", "ert_norm"])

# Plot ERT and normalized ERT by k
if not ert_df.empty:
    plt.figure(figsize=(8, 5))
    plt.plot(ert_df["k"], ert_df["ert"], marker="o", color="#4C78A8", label="ERT")
    plt.xscale("log")
    plt.title("Expected Return Time (ERT) by k")
    plt.xlabel("Occurrence Index k")
    plt.ylabel("ERT (Δ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/ert_by_k.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(ert_df["k"], ert_df["ert_norm"], marker="o", color="#F58518", label="ERT normalized")
    plt.xscale("log")
    plt.title("Normalized ERT by k")
    plt.xlabel("Occurrence Index k")
    plt.ylabel("ERT (Δ / T_u)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/ert_norm_by_k.png", dpi=150)
    plt.show()
