from datarec.io.readers.transactions import read_transactions_tabular
import os
import pandas as pd

def user_precision(user_recs, user_test, k=10):
    user_recs = user_recs[user_recs['rating'] <= k]
    item_recs = set(user_recs['item_id'])
    item_test = set(user_test['item_id'])
    if len(item_recs) == 0:
        return 0.0
    return len(item_recs.intersection(item_test)) / len(item_recs)

# dataset folder
MAIN_DATA_FOLDER = 'data/'
DATASET = 'lastfm-1k'
DATASET_FOLDER = os.path.join(MAIN_DATA_FOLDER, DATASET)

# load test set
test_path = os.path.join(DATASET_FOLDER, 'test.csv')
test = read_transactions_tabular(test_path, sep=',', user_col='user_id', item_col='item_id', rating_col='rating', header=0)

# recommendation folder
MAIN_RECS_FOLDER = 'recs/'
MODEL = 'bert_alpha_beta'
RECS_FOLDER = os.path.abspath(os.path.join(MAIN_RECS_FOLDER, MODEL, 'detailed_recommendations'))
print(RECS_FOLDER)

# store per-user results across all alpha/beta
records = []

k=40

# load recommendations
for a_tenths in range(1, 10):
    row = []
    for b_tenths in range(1, 10):
        gamma_tenths = 10 - a_tenths - b_tenths
        if gamma_tenths < 0:
            continue

        alpha = a_tenths / 10
        beta = b_tenths / 10
        gamma = gamma_tenths / 10

        alpha_s = f"{alpha:.1f}"
        beta_s = f"{beta:.1f}"
        gamma_s = f"{gamma:.1f}"

        print(f"Processing alpha={alpha_s}, beta={beta_s}...")

        recs_path = os.path.join(RECS_FOLDER, f"recs_a{alpha_s}_b{beta_s}_g{gamma_s}.csv")
        if os.path.exists(recs_path):
            recs = read_transactions_tabular(recs_path, sep=',', user_col='user_id', item_col='item_id', rating_col='rank', header=0)
        else:
            print(f"Recommendations file not found for alpha={alpha}, beta={beta}. Skipping.")
            continue

        # compute metrics for each user
        precision = dict()
        for user in recs.users:
            user_recs = recs.get_user_interactions(user)
            user_test = test.get_user_interactions(user)

            up = user_precision(user_recs, user_test, k=k)
            precision[user] = up
            records.append(
                {
                    "user_id": user,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    f"precision@{k}": up,
                }
            )

        precision_values = list(precision.values())
        print(f'Average Precision@{k}: {sum(precision_values) / len(precision_values):.4f}')

# ---- Analisi cross-alpha/beta per utente ----
df = pd.DataFrame.from_records(records)
if df.empty:
    print("No records collected. Check recs paths and inputs.")
else:
    OUTPUT_FOLDER = os.path.join('.', 'results', 'performance')
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    filename = "user_performance.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    df.to_csv(filepath, index=False, sep='\t')
    print(f"Saved user precision records to {filepath}")