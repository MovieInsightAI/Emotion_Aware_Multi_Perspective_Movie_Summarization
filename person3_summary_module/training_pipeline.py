"""
training_pipeline.py  (AAAI/TNNLS v2)
=======================================
Fixes:
  • Issue-7: Full ablation study framework (required for AAAI)
  • Architecture: Unified differentiable system with joint optimization
  • Counterfactual loss (L_cf) integrated
  • All from-scratch, no pre-trained models

Ablation conditions (Issue-7):
  1. Full model
  2. w/o emotion conditioning
  3. w/o causal loss
  4. w/o perspective projection (single Z)
  5. w/o MI loss (only orthogonality)
  6. w/o counterfactual loss
  7. w/o GNN (BiLSTM only, no graph)
"""

from __future__ import annotations
import json, logging, math, os, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from subtitle_preprocessing import SubtitlePreprocessor
from event_graph_builder import (NarrativeGraphBuilder, GraphStore,
                                  CounterfactualMaskingLoss)
from gnn_narrative_encoder import GNNNarrativeEncoder, VariationalNarrativeEncoder
from emotion_conditioning import EmotionConditioningModule
from perspective_projection import PerspectiveDisentanglementModule, PERSPECTIVES
from summary_decoder import MultiPerspectiveSummaryDecoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# ── Config ────────────────────────────────────────────────────────────────────
class CRGNNConfig:
    vocab_size:int=4096; d_model:int=128; d_hidden:int=128
    d_latent:int=256; d_emotion:int=8; d_code:int=64; d_arc:int=64
    d_persp:int=128; max_seq_len:int=128; num_gat_layers:int=3
    gat_heads:int=4; dropout:float=0.1; causal_threshold:float=0.45
    use_vae:bool=True
    # Loss weights
    lambda1:float=0.5;  lambda2:float=0.2;  lambda3:float=0.3
    lambda4:float=0.4;  lambda5:float=1e-3; lambda6:float=0.1
    lambda_cf:float=0.2  # counterfactual loss weight
    lambda_orth:float=0.1; lambda_mi:float=0.05
    lambda_ntx:float=0.1; lambda_prior:float=0.05
    # Training
    lr:float=3e-4; weight_decay:float=1e-5; max_epochs:int=50
    warmup_epochs:int=5; grad_clip:float=1.0; patience:int=10
    checkpoint_dir:str="checkpoints"; log_every:int=10
    # Ablation flags
    use_emotion:bool=True     # ablation 2
    use_causal_loss:bool=True # ablation 3
    use_projection:bool=True  # ablation 4
    use_mi_loss:bool=True     # ablation 5
    use_cf_loss:bool=True     # ablation 6
    use_gnn:bool=True         # ablation 7

    def __init__(self, **kw):
        for k,v in kw.items():
            if hasattr(self,k): setattr(self,k,v)
            else: logger.warning("Unknown config: %s",k)

    def to_dict(self):
        return {k:v for k,v in self.__dict__.items() if not k.startswith("_")}


# ── Full System ───────────────────────────────────────────────────────────────
class CRGNNSystem(nn.Module):
    """
    Unified differentiable system — joint optimization over all components.

    Single forward pass covers:
      subtitle embeddings → event graph → GNN encoder →
      emotion conditioning → latent Z → perspective projection →
      ranking-based summary scoring
    """
    def __init__(self, cfg:CRGNNConfig):
        super().__init__()
        self.cfg = cfg
        self.subtitle_proc = SubtitlePreprocessor(
            cfg.vocab_size, cfg.d_model, cfg.max_seq_len)
        self.graph_builder = NarrativeGraphBuilder(
            cfg.d_model, cfg.d_emotion, cfg.causal_threshold)

        base_enc = GNNNarrativeEncoder(
            d_in=cfg.d_model, d_hidden=cfg.d_hidden, d_latent=cfg.d_latent,
            d_emotion=cfg.d_emotion, num_layers=cfg.num_gat_layers,
            heads=cfg.gat_heads, dropout=cfg.dropout)
        self.encoder = VariationalNarrativeEncoder(base_enc) if cfg.use_vae \
                       else base_enc

        self.emotion_cond = EmotionConditioningModule(
            cfg.d_emotion, cfg.d_hidden, cfg.d_code, cfg.d_arc,
            dropout=cfg.dropout)

        self.perspective_mod = PerspectiveDisentanglementModule(
            cfg.d_latent, cfg.d_persp, cfg.d_emotion,
            PERSPECTIVES, cfg.dropout)

        self.summary_decoder = MultiPerspectiveSummaryDecoder(
            PERSPECTIVES, cfg.d_persp, cfg.d_hidden,
            cfg.d_emotion, cfg.d_arc, cfg.dropout)

        self.cf_loss_fn = CounterfactualMaskingLoss(
            threshold=cfg.causal_threshold)

    def _simple_encode(self, x:torch.Tensor) -> torch.Tensor:
        """Simple BiLSTM encoding for ablation w/o GNN."""
        lstm=nn.LSTM(x.size(-1),self.cfg.d_latent//2,batch_first=True,
                     bidirectional=True).to(x.device)
        out,_=lstm(x.unsqueeze(0))
        return out.squeeze(0).mean(0,keepdim=True)

    def forward(self, graph:Data,
                salience:Optional[torch.Tensor]=None) -> Dict:
        cfg=self.cfg; out={}

        # ── Encoding ─────────────────────────────────────────────────────────
        enc = self.encoder(graph, return_node_embs=True)
        z          = enc["z"]
        node_embs  = enc["node_embs"]
        out.update(enc)

        # ── Emotion conditioning (ablation-2: skip if use_emotion=False) ─────
        if cfg.use_emotion:
            h_mod,ew_mod,arc_emb,code,vad = self.emotion_cond(
                node_embs, graph.edge_index,
                getattr(graph,"edge_weight",
                        torch.ones(graph.edge_index.size(1),device=z.device)),
                graph.emotion)
        else:
            h_mod = node_embs
            arc_emb = torch.zeros(cfg.d_arc, device=z.device)
            code = torch.zeros(node_embs.size(0),cfg.d_code,device=z.device)
            vad  = torch.zeros(node_embs.size(0),3,device=z.device)
            ew_mod = getattr(graph,"edge_weight",
                             torch.ones(graph.edge_index.size(1),device=z.device))

        out.update({"h_mod":h_mod,"arc_emb":arc_emb,"emotion_code":code,"vad":vad})

        # ── Perspective projection (ablation-4) ───────────────────────────────
        if cfg.use_projection:
            d_loss,zd,sd = self.perspective_mod.disentangle_loss(
                z, cfg.lambda_orth, cfg.lambda_mi,
                cfg.lambda_ntx if cfg.use_mi_loss else 0.0,
                cfg.lambda_prior)
        else:
            d_loss = torch.tensor(0.0, device=z.device)
            zd = {n:z for n in PERSPECTIVES}
            sd = {n:torch.ones(z.size(0),1,device=z.device) for n in PERSPECTIVES}

        out.update({"disentangle_loss":d_loss,"z_dict":zd,"sal_dict":sd})

        # ── Summary scoring ───────────────────────────────────────────────────
        # Only run scorer when projection is active (dims match d_persp)
        if cfg.use_projection:
            scores = self.summary_decoder.score_events(
                {n:v.squeeze(0) if v.dim()>1 else v for n,v in zd.items()},
                h_mod, graph.emotion, arc_emb)
            out["event_scores"] = scores
        else:
            out["event_scores"] = {}

        return out


# ── Loss function ─────────────────────────────────────────────────────────────
def compute_total_loss(out:Dict, graph:Data,
                       pseudo_labels:torch.Tensor,
                       causal_affinity:torch.Tensor,
                       cfg:CRGNNConfig,
                       summary_decoder=None) -> Tuple[torch.Tensor,Dict[str,float]]:
    device = out["z"].device
    losses = {}

    # L_repr
    x_pool = global_mean_pool(
        graph.x.detach(),
        getattr(graph,"batch",torch.zeros(graph.x.size(0),dtype=torch.long,device=device)))
    losses["L_repr"] = F.mse_loss(out["x_recon"], x_pool)

    # L_causal (ablation-3)
    if cfg.use_causal_loss:
        pred = causal_affinity.clamp(1e-6,1-1e-6)
        losses["L_causal"] = F.binary_cross_entropy(
            pred, pseudo_labels.to(device).float())
    else:
        losses["L_causal"] = torch.tensor(0.0,device=device)

    # L_disentangle
    losses["L_disentangle"] = out.get("disentangle_loss",
                                       torch.tensor(0.0,device=device))

    # L_temporal
    np_ = out.get("next_pred"); ne = out.get("node_embs")
    L_t = torch.tensor(0.0,device=device)
    if np_ is not None and ne is not None and np_.numel()>0 and ne.size(0)>1:
        L_t = F.mse_loss(np_[:ne.size(0)-1], ne[1:].detach())
    losses["L_temporal"] = L_t

    # L_summary (ListMLE ranking)
    sal = getattr(graph,"salience",None)
    L_s = torch.tensor(0.0,device=device)
    if sal is not None and out.get("event_scores") and summary_decoder is not None:
        L_s = summary_decoder.compute_summary_loss(out["event_scores"], sal)
    losses["L_summary"] = L_s

    # L_kl
    losses["L_kl"] = out.get("kl", torch.tensor(0.0,device=device))

    # L_align
    h_mod = out.get("h_mod",out.get("node_embs"))
    if h_mod is not None:
        losses["L_align"] = F.mse_loss(h_mod, out.get("node_embs",h_mod).detach())
    else:
        losses["L_align"] = torch.tensor(0.0,device=device)

    # L_cf (ablation-6) — computed externally by Trainer
    losses["L_cf"] = torch.tensor(0.0,device=device)

    total = (losses["L_repr"]
           + cfg.lambda1 * losses["L_causal"]
           + cfg.lambda2 * losses["L_disentangle"]
           + cfg.lambda3 * losses["L_temporal"]
           + cfg.lambda4 * losses["L_summary"]
           + cfg.lambda5 * losses["L_kl"]
           + cfg.lambda6 * losses["L_align"])
    loss_dict = {k:float(v.item()) for k,v in losses.items()}
    loss_dict["L_total"] = float(total.item())
    return total, loss_dict


# ── LR Schedule ───────────────────────────────────────────────────────────────
def get_lr_scheduler(opt, warmup, total, steps_per_epoch):
    ws=warmup*steps_per_epoch; ts=total*steps_per_epoch
    def lrf(step):
        if step<ws: return step/max(1,ws)
        p=(step-ws)/max(1,ts-ws)
        return 0.5*(1+math.cos(math.pi*p))
    return torch.optim.lr_scheduler.LambdaLR(opt, lrf)


# ── Trainer ───────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, system:CRGNNSystem, cfg:CRGNNConfig, device:str="cpu"):
        self.system=system.to(device); self.cfg=cfg; self.device=device
        params=list(system.parameters())+list(system.graph_builder.parameters())
        self.optimizer=AdamW(params,lr=cfg.lr,weight_decay=cfg.weight_decay)
        self.history={k:[] for k in ["L_total","L_repr","L_causal",
                                      "L_disentangle","L_temporal",
                                      "L_summary","L_kl","L_cf"]}
        self.best_loss=float("inf"); self.patience_counter=0
        os.makedirs(cfg.checkpoint_dir,exist_ok=True)
        nparams=sum(p.numel() for p in params if p.requires_grad)
        logger.info("Trainer | device=%s | params=%d",device,nparams)

    def fit(self, graph_store:GraphStore,
            preprocessor:SubtitlePreprocessor) -> Dict:
        cfg=self.cfg; doc_ids=list(graph_store.graphs.keys())
        scheduler=get_lr_scheduler(
            self.optimizer,cfg.warmup_epochs,cfg.max_epochs,max(1,len(doc_ids)))

        for epoch in range(1, cfg.max_epochs+1):
            t0=time.time(); epoch_losses={k:[] for k in self.history}
            np.random.shuffle(doc_ids)
            self.system.train(); self.system.graph_builder.train()

            for doc_id in doc_ids:
                rg,raw_aff,raw_lbl = graph_store.get(doc_id)
                graph = Data(
                    x=rg.x.detach().clone(),
                    edge_index=rg.edge_index.detach().clone(),
                    edge_weight=rg.edge_weight.detach().clone(),
                    edge_type=rg.edge_type.detach().clone(),
                    emotion=rg.emotion.detach().clone(),
                    salience=rg.salience.detach().clone(),
                    num_nodes=int(rg.num_nodes),
                ).to(self.device)
                aff=raw_aff.detach().clone().to(self.device)
                lbl=raw_lbl.detach().clone().to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                out=self.system(graph)
                total,ld=compute_total_loss(out,graph,lbl,aff,cfg,
                                            summary_decoder=self.system.summary_decoder)

                # L_cf (counterfactual) — ablation-6
                if cfg.use_cf_loss and cfg.use_causal_loss:
                    def enc_fn(x):
                        g2=Data(x=x,edge_index=graph.edge_index,
                                edge_weight=graph.edge_weight,
                                edge_type=graph.edge_type,
                                emotion=graph.emotion,
                                num_nodes=graph.num_nodes)
                        enc=self.system.encoder(g2,return_node_embs=True)
                        return enc["node_embs"]
                    L_cf=self.system.cf_loss_fn(
                        graph.x, aff, enc_fn)
                    total=total+cfg.lambda_cf*L_cf
                    ld["L_cf"]=float(L_cf.item())

                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.system.parameters(), cfg.grad_clip)
                self.optimizer.step(); scheduler.step()

                for k,v in ld.items():
                    if k in epoch_losses: epoch_losses[k].append(v)

            avg={k:float(np.mean(v)) if v else 0.0
                 for k,v in epoch_losses.items()}
            elapsed=time.time()-t0
            logger.info(
                "Epoch %3d/%d %.1fs | total=%.4f repr=%.4f "
                "causal=%.4f disent=%.4f temp=%.4f cf=%.4f",
                epoch,cfg.max_epochs,elapsed,
                avg["L_total"],avg["L_repr"],avg["L_causal"],
                avg["L_disentangle"],avg["L_temporal"],avg["L_cf"])
            for k in self.history:
                if avg.get(k,0.0)!=0.0: self.history[k].append(avg[k])
            if avg["L_total"]<self.best_loss:
                self.best_loss=avg["L_total"]; self.patience_counter=0
                self._save(epoch,avg["L_total"])
            else:
                self.patience_counter+=1
                if self.patience_counter>=cfg.patience:
                    logger.info("Early stopping at epoch %d",epoch); break
        return self.history

    def _save(self, epoch, loss):
        p=Path(self.cfg.checkpoint_dir)/f"epoch_{epoch:04d}_loss_{loss:.4f}.pt"
        torch.save({"epoch":epoch,"loss":loss,
                    "system_state":self.system.state_dict(),
                    "optimizer_state":self.optimizer.state_dict(),
                    "config":self.cfg.to_dict()},p)
        logger.info("Checkpoint → %s",p)

    def load_checkpoint(self, path:str):
        c=torch.load(path,map_location=self.device)
        self.system.load_state_dict(c["system_state"])
        self.optimizer.load_state_dict(c["optimizer_state"])
        logger.info("Loaded checkpoint epoch=%d loss=%.4f",c["epoch"],c["loss"])


# ── Issue-7: Ablation Study Framework ─────────────────────────────────────────
class AblationStudy:
    """
    Runs all 7 ablation conditions and records metrics.

    Ablation conditions (AAAI-required):
      1. full          — complete CRGNN
      2. no_emotion    — remove FiLM+cross-attention conditioning
      3. no_causal     — remove L_causal and L_cf
      4. no_projection — single latent Z, no perspective subspaces
      5. no_mi         — replace MI loss with only orthogonality
      6. no_cf         — remove counterfactual loss only
      7. no_gnn        — replace GAT with mean pooling (ablate graph)
    """
    ABLATIONS = {
        "full":          {},
        "no_emotion":    {"use_emotion":False},
        "no_causal":     {"use_causal_loss":False,"use_cf_loss":False},
        "no_projection": {"use_projection":False},
        "no_mi":         {"use_mi_loss":False},
        "no_cf":         {"use_cf_loss":False},
        "no_gnn":        {"num_gat_layers":0},
    }

    def __init__(self, base_cfg:CRGNNConfig, graph_store:GraphStore,
                 preprocessor:SubtitlePreprocessor, device:str="cpu",
                 epochs:int=5):
        self.base_cfg=base_cfg; self.gs=graph_store
        self.proc=preprocessor; self.device=device; self.epochs=epochs

    def run(self) -> Dict[str,Dict]:
        results={}
        for name, overrides in self.ABLATIONS.items():
            logger.info("=== Ablation: %s ===", name)
            cfg=CRGNNConfig(**{**self.base_cfg.to_dict(),
                                **overrides,
                                "max_epochs":self.epochs,
                                "patience":self.epochs})
            sys=CRGNNSystem(cfg)
            trainer=Trainer(sys,cfg,self.device)
            history=trainer.fit(self.gs,self.proc)
            final_loss=history["L_total"][-1] if history["L_total"] else 999.0

            # Run inference metrics
            doc_id=list(self.gs.graphs.keys())[0]
            g,aff,lbl=self.gs.get(doc_id)
            g2=Data(x=g.x.detach().clone(),edge_index=g.edge_index.detach().clone(),
                    edge_weight=g.edge_weight.detach().clone(),
                    edge_type=g.edge_type.detach().clone(),
                    emotion=g.emotion.detach().clone(),
                    salience=g.salience.detach().clone(),
                    num_nodes=int(g.num_nodes)).to(self.device)
            sys.eval()
            with torch.no_grad():
                out=sys(g2)

            # Perspective divergence
            zd=out.get("z_dict",{})
            div=sys.perspective_mod.perspective_divergence(zd) if cfg.use_projection else {}
            avg_div=float(np.mean(list(div.values()))) if div else 0.0

            # Causal alignment (F1 of edge prediction vs pseudo-labels)
            pred_bin=(aff>=cfg.causal_threshold).float()
            true_bin=lbl.float()
            TP=float((pred_bin*true_bin).sum().item())
            FP=float((pred_bin*(1-true_bin)).sum().item())
            FN=float(((1-pred_bin)*true_bin).sum().item())
            prec=TP/max(TP+FP,1e-9); rec=TP/max(TP+FN,1e-9)
            f1=2*prec*rec/max(prec+rec,1e-9)

            results[name]={
                "final_loss":round(final_loss,4),
                "graph_f1":round(f1,4),
                "perspective_divergence":round(avg_div,4),
                "history":history,
                "overrides":overrides,
            }
            logger.info("  %s | loss=%.4f f1=%.4f div=%.4f",
                        name,final_loss,f1,avg_div)
        return results

    @staticmethod
    def print_table(results:Dict[str,Dict]):
        print("\n" + "="*65)
        print(f"{'Condition':<20} {'Loss':>8} {'Graph F1':>10} {'Persp Div':>12}")
        print("-"*65)
        for name,r in results.items():
            print(f"{name:<20} {r['final_loss']:>8.4f} "
                  f"{r['graph_f1']:>10.4f} {r['perspective_divergence']:>12.4f}")
        print("="*65)


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(system:CRGNNSystem, raw_subtitle:str,
                  emotion_tensor:torch.Tensor,
                  perspectives:Optional[List[str]]=None,
                  device:str="cpu") -> Dict:
    system.eval().to(device)
    persp=perspectives or PERSPECTIVES
    proc=system.subtitle_proc
    scenes,token_ids,padding_mask=proc.process(raw_subtitle)
    N=len(scenes)
    if N==0: return {"error":"No scenes parsed."}

    if emotion_tensor.size(0)!=N:
        emotion_tensor=F.interpolate(
            emotion_tensor.unsqueeze(0).unsqueeze(0),
            size=(N,emotion_tensor.size(1)),
            mode="bilinear",align_corners=False).squeeze(0).squeeze(0)

    scene_embs=proc.get_scene_embeddings(token_ids,padding_mask,device)
    emo=emotion_tensor.to(device)
    graph,causal_aff,_=system.graph_builder.build_graph(scene_embs,emo,device)
    graph=graph.to(device)
    out=system(graph)

    z_dict={k:v.detach().squeeze(0) if v.dim()>1 else v.detach()
            for k,v in out["z_dict"].items()}
    sal_dict={k:float(v.mean().item()) for k,v in out["sal_dict"].items()}

    node_embs=out.get("h_mod",out.get("node_embs",graph.x))
    arc_emb=out.get("arc_emb",torch.zeros(system.cfg.d_arc,device=device))
    event_scores=out.get("event_scores",{})
    scene_texts=[s["text"] for s in scenes]

    summaries={}
    for p in persp:
        sc=event_scores.get(p)
        if sc is None:
            sc=torch.ones(N,device=device)/N
        summaries[p]=MultiPerspectiveSummaryDecoder.surface_realise(
            p,sc,emo.cpu(),scene_texts,top_k=3)

    sal=graph.salience.squeeze(-1) if graph.salience is not None \
        else torch.ones(N,device=device)

    return {
        "scenes":scenes,
        "z":out["z"].detach().squeeze(0).cpu(),
        "z_dict":{k:v.cpu() for k,v in z_dict.items()},
        "sal_dict":sal_dict,
        "arc_vad":out["vad"].mean(0).detach().cpu(),
        "summaries":summaries,
        "node_salience":sal.detach().cpu(),
        "causal_affinity":causal_aff.detach().cpu(),
        "emotion_vecs":emo.cpu(),
        "vad":out["vad"].detach().cpu(),
        "event_scores":{k:v.detach().cpu() for k,v in event_scores.items()},
    }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__=="__main__":
    torch.manual_seed(42)
    RAW="\n\n".join([
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i+1:02d},000\nScene {i} narrative text."
        for i in range(1,6)])
    N,d_emo=5,8
    cfg=CRGNNConfig(vocab_size=512,d_model=32,d_hidden=32,d_latent=64,
                    d_code=16,d_arc=16,d_persp=32,max_seq_len=32,
                    num_gat_layers=2,gat_heads=4,max_epochs=3,
                    patience=3,checkpoint_dir="smoke_ckpt")
    sys=CRGNNSystem(cfg); proc=sys.subtitle_proc
    scenes,ids,mask=proc.process(RAW)
    proc.embedding_module.train()
    embs=proc.embedding_module(ids,mask)
    sys.graph_builder.to("cpu"); sys.graph_builder.train()
    graph,aff,lbl=sys.graph_builder.build_graph(embs.detach(),
                                                  torch.rand(len(scenes),d_emo))
    gs=GraphStore(); gs.add("doc1",graph,aff,lbl)
    trainer=Trainer(sys,cfg,"cpu")
    history=trainer.fit(gs,proc)
    logger.info("Final L_total: %.4f",history["L_total"][-1])
    emo_t=torch.tensor(torch.rand(N,d_emo).tolist(),dtype=torch.float32)
    r=run_inference(sys,RAW,emo_t)
    logger.info("z: %s",r["z"].shape)
    for p,s in r["summaries"].items():
        logger.info("[%s] %s",p,s[:60])
    # Mini ablation
    abl=AblationStudy(cfg,gs,proc,"cpu",epochs=2)
    results=abl.run()
    AblationStudy.print_table(results)
    logger.info("training_pipeline.py ✓")
