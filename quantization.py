#!/usr/bin/env python3
"""
WarpRec Inference & Quantization Benchmarking Script
Supporta: Grid Search Post-Hoc PQ (M, Ks) + Grid Search Alpha/Beta/Gamma
"""
import os
import csv
import torch
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from warprec.utils.registry import model_registry
from warprec.data.entities import Sessions

# Import modelli custom
try:
    import warprec_recjpq
    from warprec_recjpq.utils.faiss_pq import ProductQuantizerFAISSIndexPQOPQ
except ImportError:
    print("⚠️  Modulo warprec_recjpq non trovato - caricherò PQ da warprec standard")
    from warprec.utils.quantizer import ProductQuantizerFAISSIndexPQOPQ

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

MODELS_CONFIG = {
    'SASRecBase': {
        'checkpoint': '/home/chiara/projects/warprec/experiments/lastfm1k/legacy_split_raw_jpq/lastfm1k_30000_legacy_split_raw_jpq/ray_results/objective_function_2026-03-07_06-06-08/SASRec_gts_3465f893_1_batch_size=128,dropout_prob=0.2000,embedding_size=128,epochs=10000,learning_rate=0.0100,max_seq_len=150,n_he_2026-03-07_06-06-08/checkpoint_004884/checkpoint.pt',
        'enabled': True,
        'apply_post_hoc_pq': True, # Attiva grid search su M e Ks
    },
    # Esempio per modello JPQ già addestrato
    # 'SASRecJPQMix': {
    #     'checkpoint': '...',
    #     'enabled': False,
    #     'apply_post_hoc_pq': False, # Non serve se è già JPQ, usiamo alpha/beta
    # }
}

# Parametri Grid Search Quantizzazione
M_VALUES = [8, 16, 32, 64]   # Sottospazi (deve dividere embedding_size, es. 128)
KS_VALUES = [256]            # Solitamente 256 per 8-bit per sottospazio

# Parametri Grid Search Alpha/Beta (per modelli JPQ)
ALPHA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BETA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Dataset e output
DATASET_PATH = "data/lastfm1k_30000/legacy_global_split_raw"
OUTPUT_BASE_DIR = "./inference_output"
PATH_TO_ORIGINAL_TSV = "/home/chiara/projects/warprec/data/lastfm1k_30000/userid-timestamp-artid-artname-traid-traname.tsv"
LIMIT_REC = 100

# ============================================================================

MODEL_NAME_ALIASES = {
    "SASRecBase": "SASREC_GTS",
    "SASRec": "SASREC",
    "BertJPQMix": "BERTJPQMIX",
    "SASRecJPQMix": "SASRECJPQMIX",
}

def resolve_model_registry_name(requested_name: str) -> str:
    available = list(model_registry._registry.keys())
    aliased = MODEL_NAME_ALIASES.get(requested_name, requested_name)
    if aliased in available: return aliased
    target = aliased.replace("_", "").replace("-", "").upper()
    for name in available:
        if name.replace("_", "").replace("-", "").upper() == target: return name
    raise ValueError(f"Modello '{requested_name}' non trovato. Disponibili: {available}")

def load_lastfm_metadata(tsv_path: str):
    print(f"Caricamento metadati da {tsv_path}...")
    mapping = {}
    if not os.path.exists(tsv_path): return mapping
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                mapping[parts[4]] = (parts[3], parts[5])
    return mapping

def load_legacy_split_data(data_path: str):
    schema = {'user_id': pl.Int64, 'item_id': pl.Utf8, 'rating': pl.Float64, 'timestamp': pl.Utf8}
    train_df = pl.read_csv(os.path.join(data_path, "train.csv"), has_header=False, new_columns=list(schema.keys()), schema=schema)
    test_df = pl.read_csv(os.path.join(data_path, "test.csv"), has_header=False, new_columns=list(schema.keys()), schema=schema)
    val_path = os.path.join(data_path, "val.csv")
    val_df = pl.read_csv(val_path, has_header=False, new_columns=list(schema.keys()), schema=schema) if os.path.exists(val_path) else None
    return train_df, test_df, val_df

def load_model_from_checkpoint(checkpoint_path: str, model_name: str):
    cp_file = checkpoint_path if checkpoint_path.endswith('.pt') else os.path.join(checkpoint_path, "checkpoint.pt")
    checkpoint = torch.load(cp_file, map_location='cpu')
    info = checkpoint.get('info', {})
    params = checkpoint.get('params', checkpoint.get('hyperparameters', {}))
    model_class = model_registry.get_class(resolve_model_registry_name(model_name))
    model = model_class(params=params, info=info, seed=42)
    sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
    model.load_state_dict(sd)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device, info, params

def apply_post_hoc_quantization(model, M, Ks):
    """Applica PQ e restituisce il modello con pesi approssimati ricostruiti"""
    target_layer = None
    # Cerca il layer degli embedding degli item
    for attr in ['item_embedding', 'item_embeddings', 'embeddings']:
        if hasattr(model, attr):
            target_layer = getattr(model, attr)
            break
    if not target_layer and hasattr(model, 'item_codes_layer'):
        target_layer = model.item_codes_layer.embeddings
        
    if target_layer is None:
        print("⚠️ Layer embedding non trovato!")
        return model

    vecs = target_layer.weight.data.cpu().numpy()
    d = vecs.shape[1]
    
    if d % M != 0:
        print(f"❌ Errore: d={d} non divisibile per M={M}")
        return model

    pq = ProductQuantizerFAISSIndexPQOPQ(M=M, Ks=Ks)
    pq.fit(vecs)
    codes = pq.encode_batch(vecs, batch_size=2048)
    reconstructed = pq.decode(codes)
    
    with torch.no_grad():
        target_layer.weight.copy_(torch.from_numpy(reconstructed).to(target_layer.weight.device))
    
    return model

def generate_recommendations_generic(model, sessions, test_users, device, limit=100, info=None):
    item_mapping = info.get('item_mapping', {}) if info else {}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    recommendations = {}
    
    # Semplificato: approccio manuale robusto per tutti i modelli WarpRec
    df = sessions._get_processed_data()
    max_len = getattr(model, 'max_seq_len', 150)
    
    with torch.no_grad():
        for user_id in test_users:
            user_data = df.filter(pl.col('user_id') == user_id).sort('timestamp')
            items = [item_mapping.get(i, 0) for i in user_data['item_id'].to_list()]
            
            if not items: continue
            
            # Padding/Truncating
            items = items[-max_len:]
            padded = [0] * (max_len - len(items)) + items
            
            seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            
            # Predict scores
            if hasattr(model, 'item_codes_layer'):
                hidden = model.forward(seq_tensor)
                scores = model.item_codes_layer.score_all_items(hidden[:, -1, :])
            else:
                scores = model.forward(seq_tensor)
                if scores.dim() == 3: scores = scores[:, -1, :]
            
            top_k_scores, top_k_items = torch.topk(scores[0], limit)
            
            recs = []
            for i in range(limit):
                idx = int(top_k_items[i].item())
                recs.append((reverse_item_mapping.get(idx, str(idx)), float(top_k_scores[i].item())))
            recommendations[user_id] = recs
            
    return recommendations

def compute_metrics(recommendations, test_df, k_values=[10, 40, 100]):
    test_items_per_user = test_df.group_by('user_id').agg(pl.col('item_id'))
    gt = {row['user_id']: set(row['item_id']) for row in test_items_per_user.to_dicts()}
    
    results = {}
    for k in k_values:
        met_p, met_r, met_n = [], [], []
        for u, recs in recommendations.items():
            if u not in gt: continue
            top_k = [str(r[0]) for r in recs[:k]]
            hits = len(set(top_k) & gt[u])
            met_p.append(hits / k)
            met_r.append(hits / len(gt[u]))
            
            dcg = sum([1.0/np.log2(i+2) for i, item in enumerate(top_k) if item in gt[u]])
            idcg = sum([1.0/np.log2(i+2) for i in range(min(len(gt[u]), k))])
            met_n.append(dcg/idcg if idcg > 0 else 0)
            
        results[f'P@{k}'] = np.mean(met_p)
        results[f'R@{k}'] = np.mean(met_r)
        results[f'N@{k}'] = np.mean(met_n)
    return results

def main():
    item_metadata = load_lastfm_metadata(PATH_TO_ORIGINAL_TSV)
    train_df, test_df, val_df = load_legacy_split_data(DATASET_PATH)
    test_users = sorted(test_df['user_id'].unique().to_list())
    
    # Crea Mock Sessions
    class MockSessions:
        def __init__(self, df): self._df = df
        def _get_processed_data(self): return self._df
    sessions = MockSessions(pl.concat([train_df, val_df]) if val_df is not None else train_df)
    
    all_results = []
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    for model_name, config in MODELS_CONFIG.items():
        if not config['enabled']: continue
        
        print(f"\n🚀 Caricamento modello: {model_name}")
        model, device, info, params = load_model_from_checkpoint(config['checkpoint'], model_name)
        
        # Identifica layer per reset pesi
        target_attr = None
        for attr in ['item_embedding', 'item_embeddings', 'embeddings']:
            if hasattr(model, attr): target_attr = attr; break
        if not target_attr and hasattr(model, 'item_codes_layer'): target_attr = 'item_codes_layer.embeddings'
        
        # Backup pesi originali
        if target_attr:
            if '.' in target_attr: # handle nested
                orig_weights = model.item_codes_layer.embeddings.weight.data.clone()
            else:
                orig_weights = getattr(model, target_attr).weight.data.clone()
        
        # --- CASO 1: GRID SEARCH PQ POST-HOC ---
        if config.get('apply_post_hoc_pq', False):
            for m in M_VALUES:
                for ks in KS_VALUES:
                    print(f"\n--- Testing PQ: M={m}, Ks={ks} ---")
                    # Reset
                    if '.' in target_attr: model.item_codes_layer.embeddings.weight.data.copy_(orig_weights)
                    else: getattr(model, target_attr).weight.data.copy_(orig_weights)
                    
                    model = apply_post_hoc_quantization(model, m, ks)
                    recs = generate_recommendations_generic(model, sessions, test_users, device, LIMIT_REC, info)
                    metrics = compute_metrics(recs, test_df)
                    
                    res = {'model': model_name, 'type': 'POST_HOC_PQ', 'M': m, 'Ks': ks, 'alpha': None, 'beta': None}
                    res.update(metrics)
                    all_results.append(res)
        
        # --- CASO 2: GRID SEARCH ALPHA/BETA (se JPQ) ---
        supports_ab = hasattr(model, 'item_codes_layer') and hasattr(model.item_codes_layer, 'alpha')
        if supports_ab:
            # Ripristina pesi originali (senza post-hoc pq)
            if '.' in target_attr: model.item_codes_layer.embeddings.weight.data.copy_(orig_weights)
            
            for a in ALPHA_VALUES:
                for b in BETA_VALUES:
                    if a + b > 1.0001: continue
                    print(f"--- Testing JPQ Scoring: alpha={a}, beta={b} ---")
                    model.item_codes_layer.alpha.data.fill_(a)
                    model.item_codes_layer.beta.data.fill_(b)
                    
                    recs = generate_recommendations_generic(model, sessions, test_users, device, LIMIT_REC, info)
                    metrics = compute_metrics(recs, test_df)
                    
                    res = {'model': model_name, 'type': 'JPQ_SCORING', 'M': params.get('M', 'orig'), 'Ks': params.get('Ks', 'orig'), 'alpha': a, 'beta': b}
                    res.update(metrics)
                    all_results.append(res)

    # Salvataggio Finale
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(os.path.join(OUTPUT_BASE_DIR, "benchmarking_results.csv"), index=False)
        print(f"\n✅ Risultati salvati in {OUTPUT_BASE_DIR}/benchmarking_results.csv")
        print(df_res.head(20))

if __name__ == "__main__":
    main()