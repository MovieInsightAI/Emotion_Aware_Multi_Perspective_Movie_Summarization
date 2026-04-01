"""
gnn_narrative_encoder.py
========================
Graph Attention Network (GAT) encoder that maps narrative event graphs
into a structured latent narrative space Z.

All parameters are learned from scratch.  No pre-trained GNN weights
or external corpora are used.

Architecture
------------
Input graph G = (V, E) where
  V: event nodes with features x ∈ ℝ^{d_in}
  E: causal + temporal edges with scalar weights

Encoder stack:
  1. TypedEdgeGATConv  — GAT convolution aware of edge type (causal/temporal)
  2. EmotionGatedResidual — residual update gated by emotion conditioning
  3. Hierarchical readout — scene-level + graph-level pooling → Z

Latent space Z ∈ ℝ^{d_latent} is the graph-level narrative embedding.

Loss compatibility:
  The encoder is designed to work with all five training losses defined
  in training_pipeline.py:
    L_repr   — via graph-level Z reconstruction
    L_causal — edge attention weights supervise causal predictions
    L_temporal — next-event prediction from sequential node embeddings

References
----------
Veličković et al. (2018) "Graph Attention Networks", ICLR 2018.
Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR 2022.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.utils import softmax as pyg_softmax

logger = logging.getLogger(__name__)


# ===========================================================================
# 1.  Typed-Edge GAT Convolution
# ===========================================================================
class TypedEdgeGATConv(nn.Module):
    """
    GATv2 convolution (Brody et al., 2022) extended with:
      - Per-edge-type attention bias  (causal vs temporal)
      - Edge-weight scaling of messages

    The attention coefficient for (i→j) with edge type τ:

        e_{ij} = LeakyReLU( a_τ^T [W_src h_i || W_tgt h_j] )
        α_{ij} = softmax_{j ∈ N(i)} (e_{ij}) * w_{ij}
        h'_i   = σ( Σ_j α_{ij} W_val h_j )

    Multi-head aggregation with concatenation for intermediate layers
    and averaging for the final layer.

    Parameters
    ----------
    in_channels  : int
    out_channels : int  (per head)
    heads        : int
    num_edge_types: int  (2: temporal=0, causal=1)
    dropout      : float
    concat       : bool  True → concatenate heads; False → average
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4,
                 num_edge_types: int = 2, dropout: float = 0.1,
                 concat: bool = True):
        super().__init__()
        self.heads = heads
        self.out_ch = out_channels
        self.concat = concat
        self.dropout = dropout

        # Core GATv2 convolution
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=True,
            edge_dim=1,        # edge weight as edge feature
        )

        # Per-edge-type bias injected into attention logits
        # shape: (num_edge_types, heads, 1)
        self.edge_type_bias = nn.Parameter(
            torch.zeros(num_edge_types, heads, 1))

        # Output projection (applied after multi-head concat/avg)
        out_dim = out_channels * heads if concat else out_channels
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,             # (N, in_channels)
        edge_index: torch.Tensor,    # (2, E)
        edge_weight: torch.Tensor,   # (E,)
        edge_type: torch.Tensor,     # (E,) int64 ∈ {0, 1}
    ) -> torch.Tensor:               # (N, out_dim)
        # Incorporate per-edge-type attention bias into edge features.
        # edge_type_bias: (num_edge_types, heads, 1) → average over heads → (num_edge_types,)
        # This injects a learned scalar offset per edge type (causal vs temporal),
        # making the attention weights sensitive to edge semantics.
        type_bias = self.edge_type_bias.mean(dim=1).squeeze(-1)   # (num_edge_types,)
        edge_bias = type_bias[edge_type]                           # (E,)
        # GATv2 expects edge_attr as (E, edge_dim)
        edge_attr = (edge_weight + edge_bias).unsqueeze(-1)        # (E, 1)

        h = self.gat(x, edge_index, edge_attr=edge_attr)          # (N, out_dim)
        h = self.norm(self.out_proj(h))
        return h


# ===========================================================================
# 2.  Emotion-Gated Residual Block
# ===========================================================================
class EmotionGatedResidual(nn.Module):
    """
    Residual update gated by the scene emotion vector.

        g = σ( W_emo e_i )                 ∈ (0,1)^{d}
        h'_i = g ⊙ GATout_i + (1-g) ⊙ h_prev_i

    This allows emotion signals to selectively reinforce or suppress
    narrative information in the latent node representations.

    Parameters
    ----------
    d_node    : int   Dimensionality of node features after GAT.
    d_emotion : int   Emotion vector dimensionality.
    """

    def __init__(self, d_node: int, d_emotion: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_emotion, d_node),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_node)

    def forward(
        self,
        h_new: torch.Tensor,    # (N, d_node) — GAT output
        h_prev: torch.Tensor,   # (N, d_node) — previous node features
        emotion: torch.Tensor,  # (N, d_emotion)
    ) -> torch.Tensor:          # (N, d_node)
        g = self.gate_proj(emotion)
        h = g * h_new + (1.0 - g) * h_prev
        return self.norm(h)


# ===========================================================================
# 3.  Hierarchical Graph Readout
# ===========================================================================
class HierarchicalReadout(nn.Module):
    """
    Combines scene-level and graph-level pooling for expressive
    narrative embeddings.

    Readout = MLP( [mean_pool || max_pool || add_pool] )

    The three pooling operators capture complementary statistics:
      - mean_pool: average narrative state
      - max_pool : peak salient event
      - add_pool : cumulative narrative intensity

    Parameters
    ----------
    d_node   : int  Node feature dimensionality.
    d_latent : int  Target latent space dimensionality.
    """

    def __init__(self, d_node: int, d_latent: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_node * 3, d_node * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_node * 2, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(
        self,
        x: torch.Tensor,       # (N, d_node)
        batch: torch.Tensor,   # (N,) graph assignment
    ) -> torch.Tensor:         # (B, d_latent)
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z_sum = global_add_pool(x, batch)
        z_cat = torch.cat([z_mean, z_max, z_sum], dim=-1)
        return self.mlp(z_cat)


# ===========================================================================
# 4.  Next-Event Predictor  (for L_temporal)
# ===========================================================================
class NextEventPredictor(nn.Module):
    """
    Predicts the embedding of event t+1 from the latent representation
    of event t (auto-regressive sequence auxiliary task).

    Used to compute L_temporal in the training pipeline.

    Architecture: GRU → Linear head

    Parameters
    ----------
    d_node   : int  Node feature dim.
    d_latent : int  Latent space dim.
    """

    def __init__(self, d_node: int, d_latent: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(d_node, d_latent, batch_first=True,
                          num_layers=2, dropout=dropout)
        self.head = nn.Linear(d_latent, d_node)

    def forward(
        self,
        node_seq: torch.Tensor,   # (N, d_node) — ordered event sequence
    ) -> torch.Tensor:            # (N-1, d_node) — predicted next events
        if node_seq.size(0) < 2:
            return torch.zeros(0, node_seq.size(1),
                               device=node_seq.device)
        # Add batch dim
        x = node_seq.unsqueeze(0)                              # (1, N, d)
        out, _ = self.gru(x)                                   # (1, N, d_lat)
        pred = self.head(out.squeeze(0))                       # (N, d_node)
        return pred[:-1]                                       # (N-1, d_node)


# ===========================================================================
# 5.  Masked Event Reconstructor (auxiliary task for L_temporal)
# ===========================================================================
class MaskedEventReconstructor(nn.Module):
    """
    BERT-style masked node prediction on the event sequence.

    With probability p_mask, a node's features are replaced by a
    learned [MASK] token. The reconstructor predicts the original
    features at masked positions — this encourages the encoder to
    learn contextually-grounded event representations.

    Parameters
    ----------
    d_node  : int
    p_mask  : float   Masking probability.
    """

    def __init__(self, d_node: int, p_mask: float = 0.15):
        super().__init__()
        self.p_mask = p_mask
        self.mask_token = nn.Parameter(torch.randn(d_node) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.GELU(),
            nn.Linear(d_node, d_node),
        )

    def apply_mask(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_masked : FloatTensor (N, d) — input with masked positions
        mask      : BoolTensor (N,)   — True at masked positions
        """
        N = x.size(0)
        mask = torch.bernoulli(
            torch.full((N,), self.p_mask, device=x.device)).bool()
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        return x_masked, mask

    def forward(
        self, x_encoded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct features at masked positions."""
        return self.head(x_encoded[mask])                      # (M, d_node)


# ===========================================================================
# 6.  GNN Narrative Encoder (full module)
# ===========================================================================
class GNNNarrativeEncoder(nn.Module):
    """
    Full Graph Attention Network encoder from raw event graph to
    structured latent narrative space Z.

    Encoder pipeline
    ----------------
    EventGraph (x, edge_index, edge_weight, edge_type, emotion)
         ↓
    [TypedEdgeGATConv → EmotionGatedResidual] × num_layers
         ↓
    HierarchicalReadout → Z ∈ ℝ^{d_latent}

    Additionally exposes:
      - node_embeddings : (N, d_hidden) — for perspective projection
      - next_event_predictor : for L_temporal
      - masked_reconstructor : for L_temporal (BERT-style)

    Parameters
    ----------
    d_in       : int   Input node feature dim.
    d_hidden   : int   Hidden dim per GAT layer.
    d_latent   : int   Graph-level latent dim Z.
    d_emotion  : int   Emotion vector dim.
    num_layers : int   Number of GAT+residual blocks.
    heads      : int   Number of attention heads.
    dropout    : float
    """

    def __init__(
        self,
        d_in: int = 128,
        d_hidden: int = 128,
        d_latent: int = 256,
        d_emotion: int = 8,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
        )

        # GAT + Emotion-gated residual layers
        self.gat_layers = nn.ModuleList()
        self.emo_gates = nn.ModuleList()

        for layer_idx in range(num_layers):
            is_last = (layer_idx == num_layers - 1)
            in_ch = d_hidden if layer_idx > 0 else d_hidden
            # Intermediate layers concatenate heads; final layer averages
            self.gat_layers.append(
                TypedEdgeGATConv(
                    in_channels=in_ch,
                    out_channels=d_hidden // heads if not is_last else d_hidden,
                    heads=heads,
                    num_edge_types=2,
                    dropout=dropout,
                    concat=not is_last,
                )
            )
            gat_out_dim = d_hidden if not is_last else d_hidden
            self.emo_gates.append(
                EmotionGatedResidual(gat_out_dim, d_emotion)
            )

        # Readout
        self.readout = HierarchicalReadout(d_hidden, d_latent, dropout)

        # Auxiliary predictors
        self.next_event_pred = NextEventPredictor(d_hidden, d_latent, dropout)
        self.masked_reconstructor = MaskedEventReconstructor(d_hidden)

        # Reconstruction head (maps Z back to node space for L_repr)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_in),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        data: Data,
        return_node_embs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full encoding forward pass.

        Parameters
        ----------
        data : PyG Data object with fields:
            x            (N, d_in)
            edge_index   (2, E)
            edge_weight  (E,)
            edge_type    (E,) int64
            emotion      (N, d_emotion)
            batch        (N,)  — graph assignment (added by PyG Batch)
        return_node_embs : bool
            If True, include per-node embeddings in output dict.

        Returns
        -------
        dict with keys:
            "z"          : FloatTensor (B, d_latent)   graph embedding
            "x_recon"    : FloatTensor (N, d_in)       reconstruction
            "next_pred"  : FloatTensor (N-1, d_node)   next event pred
            "node_embs"  : FloatTensor (N, d_hidden)   (if requested)
        """
        x = data.x
        edge_index = data.edge_index
        emotion = data.emotion

        # Handle edge attributes safely
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long,
                                    device=x.device)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        h = self.input_proj(x)                                  # (N, d_hidden)
        h_prev = h.clone()

        # GAT + emotion gate layers
        for gat, emo_gate in zip(self.gat_layers, self.emo_gates):
            h_new = gat(h, edge_index, edge_weight, edge_type)
            h = emo_gate(h_new, h_prev, emotion)
            h_prev = h

        # Graph-level latent vector
        z = self.readout(h, batch)                              # (B, d_latent)

        # Auxiliary outputs
        x_recon = self.reconstruction_head(z)                  # (B, d_in)
        next_pred = self.next_event_pred(h)                    # (N-1, d_hidden)

        out = {"z": z, "x_recon": x_recon, "next_pred": next_pred}
        if return_node_embs:
            out["node_embs"] = h
        return out

    # ------------------------------------------------------------------
    # Masked reconstruction forward (for L_temporal BERT-style task)
    # ------------------------------------------------------------------
    def masked_forward(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masked reconstruction as an auxiliary training objective.

        Returns
        -------
        reconstructed : FloatTensor (M, d_hidden) — predicted at mask
        original      : FloatTensor (M, d_hidden) — true features
        z             : FloatTensor (B, d_latent)
        """
        x = data.x
        edge_index = data.edge_index
        emotion = data.emotion

        edge_weight = getattr(data, "edge_weight",
                              torch.ones(edge_index.size(1),
                                         device=x.device))
        edge_type = getattr(data, "edge_type",
                            torch.zeros(edge_index.size(1), dtype=torch.long,
                                        device=x.device))
        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long,
                                    device=x.device))

        h = self.input_proj(x)
        h_orig = h.clone()

        # Apply masking to node features
        h_masked, mask = self.masked_reconstructor.apply_mask(h)
        h_cur = h_masked

        for gat, emo_gate in zip(self.gat_layers, self.emo_gates):
            h_new = gat(h_cur, edge_index, edge_weight, edge_type)
            h_cur = emo_gate(h_new, h_cur, emotion)

        reconstructed = self.masked_reconstructor(h_cur, mask)
        original = h_orig[mask]
        z = self.readout(h_cur, batch)
        return reconstructed, original, z


# ===========================================================================
# 7.  Variational Latent Space (optional KL regularisation)
# ===========================================================================
class VariationalNarrativeEncoder(nn.Module):
    """
    Wraps GNNNarrativeEncoder with a VAE-style reparameterisation head.

    Adds a KL divergence term to the training objective to regularise
    the latent space toward a standard normal prior.

        μ, log σ² = Linear(z_det)
        z ~ N(μ, σ²)
        KL = -0.5 Σ (1 + log σ² - μ² - σ²)

    Parameters
    ----------
    base_encoder : GNNNarrativeEncoder
    """

    def __init__(self, base_encoder: GNNNarrativeEncoder):
        super().__init__()
        self.encoder = base_encoder
        d_lat = base_encoder.d_latent
        self.mu_head = nn.Linear(d_lat, d_lat)
        self.logvar_head = nn.Linear(d_lat, d_lat)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, data: Data, return_node_embs: bool = False
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(data, return_node_embs=return_node_embs)
        z_det = out["z"]
        mu = self.mu_head(z_det)
        logvar = self.logvar_head(z_det)
        z = self.reparameterise(mu, logvar)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        out["z"] = z
        out["mu"] = mu
        out["logvar"] = logvar
        out["kl"] = kl
        return out


# ===========================================================================
# 8.  Smoke test
# ===========================================================================
if __name__ == "__main__":
    from torch_geometric.data import Data

    torch.manual_seed(42)
    N, d_in, d_emo, d_lat = 6, 128, 8, 256

    # Synthetic graph
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 3],
                                [1, 2, 3, 4, 5, 3, 5]], dtype=torch.long)
    data = Data(
        x=torch.randn(N, d_in),
        edge_index=edge_index,
        edge_weight=torch.rand(edge_index.size(1)),
        edge_type=torch.randint(0, 2, (edge_index.size(1),)),
        emotion=torch.randn(N, d_emo),
    )

    encoder = GNNNarrativeEncoder(
        d_in=d_in, d_hidden=128, d_latent=d_lat, d_emotion=d_emo,
        num_layers=3, heads=4, dropout=0.1
    )
    encoder.train()
    out = encoder(data, return_node_embs=True)

    print(f"z           : {out['z'].shape}")
    print(f"x_recon     : {out['x_recon'].shape}")
    print(f"next_pred   : {out['next_pred'].shape}")
    print(f"node_embs   : {out['node_embs'].shape}")

    # VAE wrapper
    vae_enc = VariationalNarrativeEncoder(encoder)
    out_vae = vae_enc(data)
    print(f"z (vae)     : {out_vae['z'].shape}")
    print(f"KL          : {out_vae['kl'].item():.4f}")
    print("gnn_narrative_encoder.py ✓")
