import pandas as pd

# Carica il dataset completo
path = "data/lastfm1k_30000/train.csv" # usa il file originale intero
df = pd.read_csv(path)

# 1. Ordina per timestamp (come fa GlobalTemporalSplit)
df = df.sort_values('timestamp')
total_actions = len(df)

# 2. Calcola il Test Border (10% finale)
# int(len(actions) * (1 - 0.1))
test_idx = int(total_actions * 0.9)
test_timestamp = df.iloc[test_idx]['timestamp']

# 3. Calcola il Validation Border (10% di ciò che resta)
df_before_test = df[df['timestamp'] < test_timestamp]
val_idx = int(len(df_before_test) * 0.9)
val_timestamp = df_before_test.iloc[val_idx]['timestamp']

print(f"VAL_TIMESTAMP: {val_timestamp}")
print(f"TEST_TIMESTAMP: {test_timestamp}")

# --- OPZIONALE MA RACCOMANDATO ---
# Filtra gli item che appaiono solo nel test per evitare i 'nan'
train_items = set(df_before_test['item_id'].unique())
print(f"Item nel train: {len(train_items)}")
# ---------------------------------