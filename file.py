import pandas as pd
import numpy as np
from collections import Counter

# 📍 Percorsi file
path_in = "data/lastfm1k_30000/userid-timestamp-artid-artname-traid-traname.tsv"
path_out = "data/lastfm1k_30000/lastfm_30000.csv"

# 1️⃣ Carica il file (TSV senza header)
columns = ["user_id", "timestamp", "artid", "artname", "traid", "traname"]
print("Caricamento file...")
df = pd.read_csv(path_in, sep='\t', names=columns, quoting=3, on_bad_lines='skip')

# 2️⃣ Pulizia dati mancanti
# Rimuoviamo righe dove mancano i dati fondamentali per identificare la traccia o l'utente
df = df.dropna(subset=["user_id", "artname", "traname"])

# 3️⃣ Creazione identificativo unico per l'item (Senza questo scatta il KeyError)
# Poiché molti traid sono NULL, usiamo "Artista - Canzone" come identificatore
df["item_id_str"] = df["artname"].astype(str) + " - " + df["traname"].astype(str)

# 4️⃣ Converte timestamp ISO8601 in Unix timestamp (intero)
print("Conversione timestamp...")
df["timestamp"] = pd.to_datetime(df["timestamp"]).astype('int64') // 10**9

# 5️⃣ Assegna rating (Implicit feedback = 1.0)
df["rating"] = 1.0

# 6️⃣ Mappa USER_ID in interi (Risolve l'errore del join di prima)
print("Mappatura user_id...")
user2idx = {user: idx for idx, user in enumerate(df["user_id"].unique())}
df["user_id"] = df["user_id"].map(user2idx).astype(int)

# 7️⃣ Mappa ITEM_ID in interi
print("Mappatura item_id...")
item2idx = {item: idx for idx, item in enumerate(df["item_id_str"].unique())}
df["item_id"] = df["item_id_str"].map(item2idx).astype(int)

# 8️⃣ Funzione per filtrare gli item in base alla loro frequenza
def filter_items_based_on_probs(df, n_items):
    items_counter = Counter(df["item_id"])
    items_ids, item_counts = zip(*items_counter.items())
    total = sum(item_counts)
    probs = [c / total for c in item_counts]
    
    n_to_sample = min(n_items, len(items_ids))
    np.random.seed(27)
    sampled = np.random.choice(items_ids, size=n_to_sample, replace=False, p=probs)
    keep_items = set(sampled)
    return df[df["item_id"].isin(keep_items)]

# 9️⃣ Filtra per mantenere solo 30.000 item
print(f"Filtraggio item (Totali iniziali: {df['item_id'].nunique()})...")
df_filtered = filter_items_based_on_probs(df, 30000)

# 🔟 Mantieni solo le colonne finali e salva
df_filtered = df_filtered[["user_id", "item_id", "rating", "timestamp"]]

# 11️⃣ Salva
df_filtered.to_csv(path_out, index=False)

print(f"✅ Fatto! Dataset salvato in: {path_out}")
print(f"Shape finale: {df_filtered.shape}")
print(df_filtered.head())

import pandas as pd

# Carica il file attuale
df = pd.read_csv("data/lastfm1k_30000/lastfm_30000.csv")

# Assicuriamoci che siano tutti interi (tranne il rating che è float)
df['user_id'] = df['user_id'].astype(int)
df['item_id'] = df['item_id'].astype(int)
df['timestamp'] = df['timestamp'].astype(int)

# Salva SENZA header e SENZA indice
df.to_csv("data/lastfm1k_30000/lastfm_no_header.csv", index=False, header=False)

print("✅ File 'lastfm_no_header.csv' creato con successo.")
print("Esempio prime righe:")
print(df.head(3))