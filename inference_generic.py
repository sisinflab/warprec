#!/usr/bin/env python3
"""
Script di inferenza generico per modelli WarpRec.
Supporta: BertJPQMix, BertJPQ, SASRecJPQMix, gSASRecJPQ, ecc.
"""
import os
import csv
import sys
import torch
import pandas as pd
import polars as pl
from pathlib import Path

from warprec.utils.registry import model_registry
from warprec.data.entities import Sessions

# Import modelli custom per registrarli nel registry
try:
    import warprec_recjpq
except ImportError:
    print("⚠️  Modulo warprec_recjpq non trovato - alcuni modelli potrebbero non essere disponibili")


# ============================================================================
# CONFIGURAZIONE - MODIFICA QUI I TUOI CHECKPOINT
# ============================================================================

MODELS_CONFIG = {
    'SASRecJPQMix': {
        'checkpoint': '/home/chiara/projects/warprec/experiments/lastfm1k/legacy_split_raw_jpq/lastfm1k_30000_legacy_split_raw_jpq/ray_results/objective_function_2026-03-11_16-24-25/SASRecJPQMix_95101655_0_batch_size=128,centroid_strategy=svd,dropout_prob=0.2000,embedding_size=128,epochs=10000,inner_size=128,le_2026-03-11_16-24-25/checkpoint_002196/checkpoint.pt',
        'enabled': True,
    },
#     'BERT4RecJPQ': {
#         'checkpoint': 'experiments/lastfm1k/legacy_split_raw_jpq/.../checkpoint_xxx',
#         'enabled': False,  # Cambia a True quando hai il checkpoint
#     },
#     'SASRecJPQMix': {
#         'checkpoint': 'experiments/lastfm1k/legacy_split_raw_jpq/.../checkpoint_xxx',
#         'enabled': False,
#     },
#     'gSASRecJPQ': {
#         'checkpoint': 'experiments/lastfm1k/legacy_split_raw_jpq/.../checkpoint_xxx',
#         'enabled': False,
#     },
}

# Dataset e output
DATASET_PATH = "data/lastfm1k_30000/legacy_global_split_raw"
OUTPUT_BASE_DIR = "./inference_output"
PATH_TO_ORIGINAL_TSV = "/home/chiara/projects/warprec/data/lastfm1k_30000/userid-timestamp-artid-artname-traid-traname.tsv"
LIMIT_REC = 100

# ============================================================================


def load_lastfm_metadata(tsv_path: str):
    """Carica mapping track_id -> (artist_name, track_name)"""
    print(f"Caricamento metadati da {tsv_path}...")
    mapping = {}
    if not os.path.exists(tsv_path):
        print(f"ERRORE: File {tsv_path} non trovato!")
        return mapping

    count = 0
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                track_id = parts[4]
                artist_name = parts[3]
                track_name = parts[5]
                if track_id and track_id not in mapping:
                    mapping[track_id] = (artist_name, track_name)
                    count += 1

    print(f"Mappatura completata: {count} brani caricati.")
    return mapping


def load_legacy_split_data(data_path: str):
    """
    Carica il dataset dal formato legacy split usando polars.
    Returns: train_df, test_df, val_df (polars DataFrames)
    """
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")
    val_path = os.path.join(data_path, "val.csv")
    
    print(f"Caricamento split da: {data_path}")
    
    # Schema per legacy split
    schema = {
        'user_id': pl.Int64,
        'item_id': pl.Utf8,
        'rating': pl.Float64,
        'timestamp': pl.Utf8  # ISO datetime string format
    }
    
    train_df = pl.read_csv(train_path, has_header=False, new_columns=list(schema.keys()), schema=schema)
    test_df = pl.read_csv(test_path, has_header=False, new_columns=list(schema.keys()), schema=schema)
    
    val_df = None
    if os.path.exists(val_path):
        val_df = pl.read_csv(val_path, has_header=False, new_columns=list(schema.keys()), schema=schema)
        print(f"Val: {len(val_df)} interazioni, {val_df['user_id'].n_unique()} utenti")
    
    print(f"Train: {len(train_df)} interazioni, {train_df['user_id'].n_unique()} utenti")
    print(f"Test: {len(test_df)} interazioni, {test_df['user_id'].n_unique()} utenti")
    
    return train_df, test_df, val_df


def load_model_from_checkpoint(checkpoint_path: str, model_name: str):
    """
    Carica modello generico dal checkpoint.
    
    Args:
        checkpoint_path: Path al file checkpoint (.pt) o directory contenente checkpoint
        model_name: Nome del modello (es: 'BertJPQMix', 'BERT4RecJPQ', 'SASRecJPQMix')
    
    Returns:
        model, device
    """
    print(f"\nCaricamento modello '{model_name}' da: {checkpoint_path}")
    
    # Gestisci sia file .pt che directory
    if checkpoint_path.endswith('.pt') and os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
    else:
        # Cerca model.pt o checkpoint.pt nella directory
        checkpoint_file = os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_file}")
    
    # Carica checkpoint
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Debug: mostra chiavi disponibili
    print(f"Chiavi checkpoint: {list(checkpoint.keys())}")
    
    # Estrai info e parametri (gestisci diversi formati)
    info = checkpoint.get('info', {})
    params = checkpoint.get('params', {})
    
    # Se non ci sono params, potrebbe essere un checkpoint Ray Tune
    if not params and 'hyperparameters' in checkpoint:
        params = checkpoint['hyperparameters']
    
    print(f"Info modello: {info}")
    print(f"Parametri: {params}")
    
    # Ottieni classe modello dal registry
    try:
        model_class = model_registry.get_class(model_name)
    except (KeyError, ValueError) as e:
        available = list(model_registry._registry.keys()) if hasattr(model_registry, '_registry') else []
        raise ValueError(f"Modello '{model_name}' non trovato nel registry. Disponibili: {available}") from e
    
    # Crea modello
    model = model_class(params=params, info=info, seed=42)
    
    # Carica pesi (gestisci diversi formati di checkpoint)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume che l'intero checkpoint sia lo state_dict
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            raise ValueError(f"Impossibile caricare state_dict. Chiavi disponibili: {list(checkpoint.keys())}") from e
    
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Modello caricato su device: {device}")
    return model, device, info, params


def create_sessions_from_dataframe(train_df: pl.DataFrame, val_df: pl.DataFrame = None):
    """
    Crea oggetto Sessions dal DataFrame polars per compatibility con modelli WarpRec.
    """
    # Combina train e val se disponibile
    if val_df is not None:
        combined_df = pl.concat([train_df, val_df])
    else:
        combined_df = train_df
    
    # Crea Sessions mock (necessario per alcuni modelli)
    class MockSessions:
        def __init__(self, df):
            self._df = df
            self.user_label = 'user_id'
            self.item_label = 'item_id'
            self.timestamp_label = 'timestamp'
            self.rating_label = 'rating'
            
            # Per modelli che usano _user_offsets
            users = df['user_id'].unique().sort()
            self._user_offsets = [0]
            self._flat_items = []
            
            for user_id in users:
                user_data = df.filter(pl.col('user_id') == user_id).sort('timestamp')
                items = user_data['item_id'].to_list()
                self._flat_items.extend(items)
                self._user_offsets.append(len(self._flat_items))
        
        def _get_processed_data(self):
            return self._df
    
    return MockSessions(combined_df)


def generate_recommendations_generic(model, sessions, test_users, device, limit: int = 100, info: dict = None):
    """
    Genera raccomandazioni in modo generico per diversi tipi di modelli.
    
    Args:
        model: Modello WarpRec
        sessions: Oggetto Sessions con i dati
        test_users: Lista di user_id per cui generare raccomandazioni
        device: torch device
        limit: Numero di raccomandazioni per utente
        info: Dict con metadata del modello (item_mapping, user_mapping, ecc.)
    
    Returns:
        dict {user_id: [(item_id, score), ...]}
    """
    print(f"\nGenerazione raccomandazioni Top-{limit} per {len(test_users)} utenti...")
    
    # Estrai mappings da info se disponibili
    item_mapping = info.get('item_mapping', {}) if info else {}
    user_mapping = info.get('user_mapping', {}) if info else {}
    
    # Crea reverse mapping (indice -> item_id originale)
    reverse_item_mapping = {v: k for k, v in item_mapping.items()} if item_mapping else {}
    reverse_user_mapping = {v: k for k, v in user_mapping.items()} if user_mapping else {}
    
    recommendations = {}
    
    # Ottieni dataloader per test (se il modello supporta get_dataloader)
    if hasattr(model, 'get_dataloader'):
        try:
            # Per modelli sequenziali
            test_loader = model.get_dataloader(
                interactions=None,
                sessions=sessions,
                mode='test',
                shuffle=False
            )
            
            with torch.no_grad():
                for batch in test_loader:
                    # Estrai user IDs dal batch se disponibili
                    if 'seq' in batch:
                        # Modelli sequenziali (BERT, SASRec, ecc.)
                        seq = batch['seq'].to(device)
                        attn = batch.get('attn', None)
                        if attn is not None:
                            attn = attn.to(device)
                        
                        # Ottieni scores
                        if hasattr(model, 'recommend_impl'):
                            # Usa recommend_impl se disponibile
                            result = model.recommend_impl(batch, limit, 'test')
                            items = result.get('items', result.get('top_k_items'))
                            scores = result.get('scores', result.get('top_k_scores'))
                            
                            # Map to user IDs (assume batch order matches user order)
                            for i, user_id in enumerate(test_users[:len(items)]):
                                user_items = items[i].cpu().tolist() if torch.is_tensor(items[i]) else items[i]
                                user_scores = scores[i].cpu().tolist() if torch.is_tensor(scores[i]) else scores[i]
                                recommendations[user_id] = list(zip(user_items, user_scores))
                        else:
                            # Fallback: usa forward + score_all_items
                            hidden = model.forward(seq, attention_mask=attn)
                            if hasattr(model, 'item_codes_layer'):
                                # Ottieni ultimo hidden state
                                if hidden.dim() == 3:
                                    hidden = hidden[:, -1, :]
                                scores = model.item_codes_layer.score_all_items(hidden)
                            else:
                                scores = hidden
                            
                            top_k_scores, top_k_items = torch.topk(scores, limit, dim=1)
                            
                            for i, user_id in enumerate(test_users[:len(top_k_items)]):
                                user_items = top_k_items[i].cpu().tolist()
                                user_scores = top_k_scores[i].cpu().tolist()
                                recommendations[user_id] = list(zip(user_items, user_scores))
        except Exception as e:
            print(f"Errore con get_dataloader: {e}")
            print("Tentativo con approccio manuale...")
    
    # Fallback: approccio manuale per modelli che non supportano get_dataloader
    if not recommendations:
        print("Usando approccio manuale per generazione raccomandazioni...")
        
        # Costruisci sequenze utente dal sessions
        df = sessions._get_processed_data()
        
        with torch.no_grad():
            for user_id in test_users:
                # Ottieni sequenza utente
                user_data = df.filter(pl.col('user_id') == user_id).sort('timestamp')
                
                if len(user_data) == 0:
                    recommendations[user_id] = [(0, 0.0)] * limit
                    continue
                
                items = user_data['item_id'].to_list()
                
                # Converti item_id in indici se necessario
                if item_mapping and items and isinstance(items[0], str):
                    items_indices = [item_mapping.get(item, 0) for item in items]
                else:
                    items_indices = items
                
                # Prepara input in base al tipo di modello
                if hasattr(model, 'max_seq_len'):
                    # Modello sequenziale
                    max_len = model.max_seq_len
                    
                    if len(items_indices) > max_len:
                        items_indices = items_indices[-max_len:]
                    
                    # Gestisci padding diversamente per BERT (con mask) vs SASRec (senza mask)
                    if hasattr(model, 'mask_token_id'):
                        # BERT4Rec style: aggiungi mask token alla fine
                        if hasattr(model, 'padding_token_id'):
                            padded = [model.padding_token_id] * (max_len - len(items_indices) - 1) + items_indices + [model.mask_token_id]
                        else:
                            padded = [0] * (max_len - len(items_indices) - 1) + items_indices + [0]
                    else:
                        # SASRec style: no mask token, solo padding
                        if hasattr(model, 'padding_token_id'):
                            padded = [model.padding_token_id] * (max_len - len(items_indices)) + items_indices
                        else:
                            padded = [0] * (max_len - len(items_indices)) + items_indices
                    
                    seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
                    seq_len = torch.tensor([len(items)], dtype=torch.long, device=device)
                    user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
                    
                    # Predict
                    try:
                        if hasattr(model, 'predict'):
                            scores = model.predict(
                                user_indices=user_tensor,
                                user_seq=seq_tensor,
                                seq_len=seq_len
                            )
                        else:
                            scores = model.forward(seq_tensor)
                            if scores.dim() == 3:
                                scores = scores[:, -1, :]
                            if hasattr(model, 'item_codes_layer'):
                                scores = model.item_codes_layer.score_all_items(scores)
                        
                        top_k_scores, top_k_items = torch.topk(scores[0], limit)
                        
                        # Converti indici in item_id originali se necessario
                        if reverse_item_mapping:
                            recs = [
                                (reverse_item_mapping.get(int(top_k_items[i].item()), str(top_k_items[i].item())), 
                                 float(top_k_scores[i].item()))
                                for i in range(limit)
                            ]
                        else:
                            recs = [
                                (int(top_k_items[i].item()), float(top_k_scores[i].item()))
                                for i in range(limit)
                            ]
                        recommendations[user_id] = recs
                    except Exception as e:
                        print(f"Errore per utente {user_id}: {e}")
                        recommendations[user_id] = [(0, 0.0)] * limit
    
    print(f"Raccomandazioni generate per {len(recommendations)} utenti")
    return recommendations


def save_recommendations_csv(recommendations: dict, item_metadata: dict, output_path: str):
    """Salva raccomandazioni in CSV con metadati"""
    print(f"\nSalvataggio raccomandazioni in: {output_path}")
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "rank", "item_id", "artist_name", "track_name", "score"])
        
        for user_id in sorted(recommendations.keys()):
            user_recs = recommendations[user_id]
            
            for rank, (item_id, score) in enumerate(user_recs, start=1):
                item_id_str = str(item_id)
                artist, track = item_metadata.get(item_id_str, (item_id_str, "Unknown Track"))
                writer.writerow([user_id, rank, item_id_str, artist, track, score])
    
    print(f"Salvate {sum(len(r) for r in recommendations.values())} raccomandazioni")


def compute_metrics(recommendations: dict, test_df: pl.DataFrame, k_values=[10, 40, 100]):
    """Calcola Precision, Recall, nDCG per vari k"""
    print("\nCalcolo metriche...")
    
    # Ground truth per utente
    test_items_per_user = {}
    for user_id in test_df['user_id'].unique():
        user_items = test_df.filter(pl.col('user_id') == user_id)['item_id'].to_list()
        test_items_per_user[int(user_id)] = set(user_items)
    
    results = {}
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id, recs in recommendations.items():
            if user_id not in test_items_per_user:
                continue
            
            top_k_items = [str(item_id) for item_id, _ in recs[:k]]
            relevant_items = test_items_per_user[user_id]
            
            hits = len(set(top_k_items) & relevant_items)
            
            precision = hits / k if k > 0 else 0
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
            
            # Calcola nDCG
            dcg = 0.0
            for i, item_id in enumerate(top_k_items):
                if item_id in relevant_items:
                    dcg += 1.0 / (i + 2)  # i+2 perché i parte da 0, e log2(1) = 0
            
            # IDCG (Ideal DCG)
            idcg = sum(1.0 / (i + 2) for i in range(min(len(relevant_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
        
        results[f'Precision@{k}'] = sum(precisions) / len(precisions) if precisions else 0
        results[f'Recall@{k}'] = sum(recalls) / len(recalls) if recalls else 0
        results[f'nDCG@{k}'] = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    
    return results


def main():
    print("=" * 80)
    print("INFERENZA MODELLI WARPREC - GRID SEARCH ALPHA/BETA")
    print("=" * 80)
    
    # Parametri per grid search
    alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    beta_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 1. Carica metadati (una volta sola)
    item_metadata = load_lastfm_metadata(PATH_TO_ORIGINAL_TSV)
    
    # 2. Carica dati (una volta sola)
    train_df, test_df, val_df = load_legacy_split_data(DATASET_PATH)
    test_users = sorted(test_df['user_id'].unique().to_list())
    print(f"\nUtenti test totali: {len(test_users)}")
    
    # 3. Crea Sessions object (una volta sola)
    sessions = create_sessions_from_dataframe(train_df, val_df)
    
    # 4. Processa ogni modello abilitato con grid search su alpha/beta
    all_results = []
    
    for model_name, config in MODELS_CONFIG.items():
        if not config['enabled']:
            print(f"\n{'='*80}")
            print(f"⏭️  SKIP: {model_name} (disabilitato)")
            print(f"{'='*80}")
            continue
        
        checkpoint_path = config['checkpoint']
        
        print(f"\n{'='*80}")
        print(f"📊 PROCESSING: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Setup output per questo modello
            output_dir = os.path.join(OUTPUT_BASE_DIR, model_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Carica modello (una sola volta)
            model, device, info, params = load_model_from_checkpoint(checkpoint_path, model_name)
            
            # Verifica che il modello abbia item_codes_layer con alpha/beta
            if not hasattr(model, 'item_codes_layer'):
                print(f"⚠️  {model_name} non ha 'item_codes_layer', salto grid search")
                continue
            
            if not hasattr(model.item_codes_layer, 'alpha') or not hasattr(model.item_codes_layer, 'beta'):
                print(f"⚠️  {model_name}.item_codes_layer non ha parametri alpha/beta, salto grid search")
                continue
            
            print(f"\n🔍 Grid Search su alpha/beta per {model_name}...")
            print(f"   Alpha values: {alpha_values}")
            print(f"   Beta values: {beta_values}")
            
            # Grid search su alpha/beta
            for alpha in alpha_values:
                for beta in beta_values:
                    # Salta combinazioni invalide
                    if alpha + beta > 1.0:
                        continue
                    
                    gamma = 1.0 - alpha - beta
                    if gamma < 0.0:
                        continue
                    
                    run_name = f"{model_name}_alpha{alpha:.2f}_beta{beta:.2f}_gamma{gamma:.2f}"
                    print(f"\n{'='*60}")
                    print(f"🔧 {run_name}")
                    print(f"{'='*60}")
                    
                    # Setta alpha/beta nel modello
                    model.item_codes_layer.alpha.data.fill_(alpha)
                    model.item_codes_layer.beta.data.fill_(beta)
                    
                    # Genera raccomandazioni
                    recommendations = generate_recommendations_generic(
                        model, sessions, test_users, device, limit=LIMIT_REC, info=info
                    )
                    
                    # Salva CSV dettagliato per ogni combinazione alpha/beta/gamma
                    recs_dir = os.path.join(output_dir, "detailed_recommendations")
                    os.makedirs(recs_dir, exist_ok=True)
                    out_csv = os.path.join(recs_dir, f"recs_a{alpha:.1f}_b{beta:.1f}_g{gamma:.1f}.csv")
                    save_recommendations_csv(recommendations, item_metadata, out_csv)
                    
                    # Calcola metriche
                    metrics = compute_metrics(recommendations, test_df, k_values=[10, 40, 100])
                    
                    # Store risultati
                    result_row = {
                        'model': model_name,
                        'run_name': run_name,
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                    }
                    result_row.update(metrics)
                    all_results.append(result_row)
                    
                    print(f"✅ Metriche:")
                    for metric, value in metrics.items():
                        print(f"   {metric:20s}: {value:.4f}")
            
            print(f"\n{'='*80}")
            print(f"✅ COMPLETATO: {model_name} - {len([r for r in all_results if r['model'] == model_name])} configurazioni testate")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\n❌ ERRORE per {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. Salva tutti i risultati
    if all_results:
        print("\n" + "=" * 80)
        print("📊 SALVATAGGIO RISULTATI")
        print("=" * 80)
        
        # Crea DataFrame con tutti i risultati
        df_results = pd.DataFrame(all_results)
        
        # Salva CSV completo
        results_csv = os.path.join(OUTPUT_BASE_DIR, "all_alpha_beta_results.csv")
        df_results.to_csv(results_csv, index=False)
        print(f"\n✅ Tutti i risultati salvati in: {results_csv}")
        
        # Mostra top 10 configurazioni per nDCG@10
        print("\n🏆 Top 10 configurazioni per nDCG@10:")
        top10 = df_results.nlargest(10, 'nDCG@10')[['run_name', 'alpha', 'beta', 'gamma', 'nDCG@10', 'nDCG@40', 'nDCG@100', 'Precision@40']]
        print(top10.to_string(index=False))
        
        print("\n" + "=" * 80)
    else:
        print("\n⚠️  Nessuna configurazione testata con successo.")


if __name__ == "__main__":
    main()
