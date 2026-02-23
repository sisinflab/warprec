# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235, R0902
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry

# Assicurati che questo import punti correttamente al tuo file nel progetto
from  ..recjpq.rec_jpq_layer import ItemCodeLayer

@model_registry.register(name="BERT4RecJPQ")
class BERT4RecJPQ(IterativeRecommender, SequentialRecommenderUtils):
    """
    BERT4Rec integrato con Joint Product Quantization (JPQ).
    L'inizializzazione SVD avviene in modo del tutto automatico al primo caricamento del dataloader.
    """

    DATALOADER_TYPE = DataLoaderType.CLOZE_MASK_LOADER

    # Model hyperparameters (standard BERT4Rec)
    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    mask_prob: float
    batch_size: int
    epochs: int
    learning_rate: float
    max_seq_len: int
    
    # Parametri aggiuntivi per JPQ (vengono presi dai config di WarpRec)
    pq_m: int
    centroid_strategy: str

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        self.padding_token_id = self.n_items
        self.mask_token_id = self.n_items + 1

        # Fallback sui default se non specificati nei config
        self.pq_m = getattr(self, 'pq_m', 4)
        self.centroid_strategy = getattr(self, 'centroid_strategy', 'svd')

        # Layer JPQ
        self.item_codes_layer = ItemCodeLayer(
            embedding_size=self.embedding_size,
            pq_m=self.pq_m,                  
            num_items=self.n_items,          
            sequence_length=self.max_seq_len,
            codes_strategy=self.centroid_strategy
        )

        self.position_embedding = nn.Embedding(self.max_seq_len + 1, self.embedding_size)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
        self.dropout = nn.Dropout(self.dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # Flag per l'inizializzazione SVD automatica
        self._centroids_assigned = False

        self.apply(self._init_weights)

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs):
        """
        Intercettiamo il dataloader per estrarre i dati di train ed eseguire l'SVD
        prima che parta la backpropagation.
        """
        if not self._centroids_assigned:
            print("\n[BERT4Rec-JPQ] Avvio inizializzazione automatica SVD Centroids...")
            
            # --- TENTATIVO 1: Accesso diretto a 'sessions' ---
            # Questo è il percorso più veloce se sessions ha già fatto il lavoro sporco
            # Controlliamo l'attributo privato _cached_user_histories che ho visto nel tuo codice Sessions
            raw_histories = getattr(sessions, '_cached_user_histories', None)
            
            if raw_histories and len(raw_histories) > 0:
                print("[BERT4Rec-JPQ] Sequenze trovate in '_cached_user_histories'!")
                train_users = []
                for uid, items in raw_histories.items():
                    # Formato richiesto da SVD: lista di tuple (0, item_id)
                    train_users.append([(0, int(i)) for i in items])
            
            else:
                # --- TENTATIVO 2: Ricostruzione dalla Matrice Sparsa di Interactions ---
                print("[BERT4Rec-JPQ] Sequenze non trovate in cache. Ricostruzione da 'Interactions.get_sparse()'...")
                
                # Otteniamo la matrice sparsa (CSR format)
                # Righe = Utenti Mappati, Colonne = Item Mappati
                sparse_mat = interactions.get_sparse()
                
                train_users = []
                n_users = sparse_mat.shape[0]
                
                # Iteriamo sulle righe della matrice sparsa. 
                # È molto veloce perché accediamo agli array numpy interni (indptr, indices)
                for u in range(n_users):
                    start = sparse_mat.indptr[u]
                    end = sparse_mat.indptr[u+1]
                    
                    if end > start: # Se l'utente ha interazioni
                        # Gli indici delle colonne sono gli item ID mappati
                        user_items = sparse_mat.indices[start:end]
                        
                        # Creiamo la lista di tuple [(0, item_id), ...]
                        # Nota: SVD si aspetta che gli item siano ordinati temporalmente?
                        # La matrice sparsa CSR perde l'ordine temporale se non è stata costruita ordinata.
                        # Tuttavia, per l'SVD (Co-occurrence) l'ordine esatto NON è critico, basta l'insieme degli item.
                        train_users.append([(0, int(item)) for item in user_items])

            # Controllo finale
            if len(train_users) == 0:
                raise ValueError("❌ [BERT4Rec-JPQ] Errore: 0 sequenze estratte. Impossibile calcolare SVD.")

            print(f"[BERT4Rec-JPQ] Estratte {len(train_users)} sequenze utente per SVD.")
            
            # Eseguiamo la tua assegnazione
            self.item_codes_layer.assign_codes(train_users)
            self._centroids_assigned = True
            print("[BERT4Rec-JPQ] Inizializzazione SVD completata con successo.\n")

        return sessions.get_cloze_mask_dataloader(
            max_seq_len=self.max_seq_len,
            mask_prob=self.mask_prob,
            neg_samples=1, 
            batch_size=self.batch_size,
            mask_token_id=self.mask_token_id,
            **kwargs,
        )
    def forward(self, item_seq: Tensor) -> Tensor:
        padding_mask = item_seq == self.padding_token_id

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_codes_layer(item_seq)
        pos_emb = self.position_embedding(position_ids)

        input_emb = self.layernorm(item_emb + pos_emb)
        input_emb = self.dropout(input_emb)

        transformer_output = self.transformer_encoder(
            src=input_emb, mask=None, src_key_padding_mask=padding_mask
        )
        return transformer_output

    def train_step(self, batch: Any, *args, **kwargs):
        """
        CrossEntropy calcolata unicamente sui token mascherati per velocizzare
        enormemente il training e risparmiare VRAM.
        """
        masked_seq, pos_items, neg_items, masked_indices = batch

        transformer_output = self.forward(masked_seq)

        # masked_indices ha forma [batch_size, num_masked_positions]
        # Estraiamo i vettori dal transformer in quelle esatte posizioni
        seq_output = self._multi_hot_gather(transformer_output, masked_indices) # [Batch, NumMasks, Emb]
        
        # Filtriamo il padding all'interno della maschera di WarpRec
        loss_mask = masked_indices > 0
        seq_output_masked = seq_output[loss_mask] # Appiattisce in [N_Masked_Tokens, Emb]
        labels_masked = pos_items[loss_mask]      # Appiattisce in [N_Masked_Tokens]

        if seq_output_masked.size(0) == 0:
            return torch.tensor(0.0, device=transformer_output.device, requires_grad=True)

        # Calcoliamo i punteggi su tutti i 29k item MA SOLO per i token che partecipano alla loss
        logits = self.item_codes_layer.score_masked_tokens(seq_output_masked)

        # Loss nativa velocissima di PyTorch
        loss = F.cross_entropy(logits, labels_masked, ignore_index=-100)

        return loss

    def _multi_hot_gather(self, source: Tensor, indices: Tensor) -> Tensor:
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, source.size(-1))
        return torch.gather(source, 1, indices_expanded)

    def _prepare_for_prediction(self, user_seq: Tensor, seq_len: Tensor) -> Tensor:
        pred_seq = torch.full(
            (user_seq.size(0), user_seq.size(1) + 1),
            self.padding_token_id,
            dtype=torch.long,
            device=user_seq.device,
        )
        pred_seq[:, : user_seq.size(1)] = user_seq
        batch_indices = torch.arange(user_seq.size(0), device=user_seq.device)
        pred_seq[batch_indices, seq_len] = self.mask_token_id
        return pred_seq

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        user_seq: Optional[Tensor] = None,
        seq_len: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        
        pred_seq = self._prepare_for_prediction(user_seq, seq_len)
        transformer_output = self.forward(pred_seq)
        
        # Estrae l'embedding associato all'ultima posizione aggiunta (il MASK)
        seq_output = self._gather_indexes(transformer_output, seq_len)

        if item_indices is None:
            # Full rank: restituisce score verso tutti gli items
            predictions = self.item_codes_layer.score_all_items(seq_output)
        else:
            # Sampled rank: score solo per un set specifico (es. 100 negativi)
            predictions = self.item_codes_layer.score_sequence_items(seq_output, item_indices)

        return predictions