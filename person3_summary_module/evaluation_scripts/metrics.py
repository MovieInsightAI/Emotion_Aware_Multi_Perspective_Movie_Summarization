"""
evaluation_scripts/metrics.py  (AAAI/TNNLS v2)
================================================
Full evaluation suite + ablation study reporting.
All metrics computed from learned representations only.
"""

from __future__ import annotations
import math, logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── ROUGE ─────────────────────────────────────────────────────────────────────
def _toks(text): 
    import re; return re.findall(r"[a-z]+",text.lower())

def _ngrams(tokens,n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _lcs(a,b):
    m,n=len(a),len(b); dp=[[0]*(n+1) for _ in range(2)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i%2][j]=dp[(i-1)%2][j-1]+1 if a[i-1]==b[j-1] \
                        else max(dp[(i-1)%2][j],dp[i%2][j-1])
    return dp[m%2][n]

def rouge_n(hyp,ref,n=1):
    h=_ngrams(_toks(hyp),n); r=_ngrams(_toks(ref),n)
    if not r: return 0.0
    ov=sum((h&r).values())
    rec=ov/max(sum(r.values()),1); prec=ov/max(sum(h.values()),1)
    return 2*prec*rec/max(prec+rec,1e-9)

def rouge_l(hyp,ref):
    ht,rt=_toks(hyp),_toks(ref)
    if not rt or not ht: return 0.0
    lcs=_lcs(ht,rt)
    prec=lcs/len(ht); rec=lcs/len(rt)
    return 2*prec*rec/max(prec+rec,1e-9)

def compute_rouge_scores(hyps,refs):
    r1,r2,rl=[],[],[]
    for h,r in zip(hyps,refs):
        r1.append(rouge_n(h,r,1)); r2.append(rouge_n(h,r,2)); rl.append(rouge_l(h,r))
    return {"rouge1":float(np.mean(r1)),"rouge2":float(np.mean(r2)),
            "rougeL":float(np.mean(rl))}


# ── BLEU ──────────────────────────────────────────────────────────────────────
def bleu_score(hyp,ref,max_n=4):
    ht,rt=_toks(hyp),_toks(ref)
    if not ht: return {f"bleu{n}":0.0 for n in range(1,max_n+1)}
    scores={}; log_geo=0.0
    for n in range(1,max_n+1):
        hn=_ngrams(ht,n); rn=_ngrams(rt,n)
        clipped=sum(min(c,rn.get(g,0)) for g,c in hn.items())
        p=clipped/max(sum(hn.values()),1)
        scores[f"bleu{n}"]=p; log_geo+=(1/max_n)*math.log(max(p,1e-10))
    bp=min(1.0,math.exp(1-len(rt)/max(len(ht),1)))
    scores["bleu_cumulative"]=bp*math.exp(log_geo)
    return scores

def compute_bleu_scores(hyps,refs):
    acc={}
    for h,r in zip(hyps,refs):
        for k,v in bleu_score(h,r).items():
            acc.setdefault(k,[]).append(v)
    return {k:float(np.mean(v)) for k,v in acc.items()}


# ── Embedding-based similarity ────────────────────────────────────────────────
def embedding_similarity(e1,e2):
    if e1.dim()==1: e1=e1.unsqueeze(0)
    if e2.dim()==1: e2=e2.unsqueeze(0)
    return float(F.cosine_similarity(e1,e2,dim=-1).mean().item())


# ── Graph alignment ───────────────────────────────────────────────────────────
def graph_alignment_score(pred_adj,true_adj,threshold=0.5):
    pb=(pred_adj>=threshold).float(); tb=true_adj.float()
    TP=float((pb*tb).sum()); FP=float((pb*(1-tb)).sum())
    FN=float(((1-pb)*tb).sum())
    j=TP/max(TP+FP+FN,1e-9)
    prec=TP/max(TP+FP,1e-9); rec=TP/max(TP+FN,1e-9)
    f1=2*prec*rec/max(prec+rec,1e-9)
    sca=(TP-FP)/max(TP+FN+FP,1e-9)
    return {"graph_jaccard":float(j),"graph_precision":float(prec),
            "graph_recall":float(rec),"graph_f1":float(f1),"sca":float(sca)}


# ── Perspective divergence ────────────────────────────────────────────────────
def perspective_divergence(z_dict):
    keys=list(z_dict.keys()); div={}
    for i,k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            d=(z_dict[k1].flatten()-z_dict[k2].flatten()).norm().item()
            div[f"{k1}↔{k2}"]=float(d)
    return div


# ── Emotion consistency ───────────────────────────────────────────────────────
def emotion_consistency(vad_pred,vad_true=None):
    out={}
    if vad_pred.size(0)>1:
        out["arc_smoothness"]=float((vad_pred[1:]-vad_pred[:-1]).abs().mean().item())
    if vad_pred.size(1)>0:
        v=vad_pred[:,0]
        out["valence_span"]=float((v.max()-v.min()).item())
    if vad_true is not None:
        out["emotion_consistency"]=float(F.cosine_similarity(
            vad_pred.flatten().unsqueeze(0),
            vad_true.flatten().unsqueeze(0)).item())
    return out


# ── Latent space quality ──────────────────────────────────────────────────────
def latent_space_quality(z_list):
    if not z_list: return {}
    Z=torch.stack(z_list); norms=Z.norm(dim=-1)
    M=Z.size(0); idx=torch.randperm(M)[:min(M,100)]
    dists=[(Z[idx[i]]-Z[idx[i+1]]).norm().item()
           for i in range(0,len(idx)-1,2)]
    try:
        sv=torch.linalg.svdvals(Z-Z.mean(0))
        iso=float(sv.min()/max(sv.max(),1e-9))
    except Exception: iso=0.0
    return {"mean_norm":float(norms.mean()),"std_norm":float(norms.std()),
            "mean_pairwise_d":float(np.mean(dists)) if dists else 0.0,
            "isotropy":iso}


# ── Issue-7: Ablation Table Formatter ────────────────────────────────────────
def format_ablation_table(results:Dict[str,Dict]) -> str:
    """
    Format ablation results as a LaTeX-compatible table string
    suitable for AAAI/TNNLS submission.
    """
    lines=[
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Ablation Study — CRGNN Components}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Condition & L\\_total↓ & Graph F1↑ & Persp Div↑ \\\\",
        "\\midrule",
    ]
    for name,r in results.items():
        tag = "\\textbf{" + name + "}" if name=="full" else name
        lines.append(
            f"{tag} & {r['final_loss']:.4f} & "
            f"{r['graph_f1']:.4f} & "
            f"{r['perspective_divergence']:.4f} \\\\")
    lines += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    return "\n".join(lines)


def format_results(results):
    lines=["="*60,"Evaluation Results","="*60]
    for g,metrics in results.items():
        lines.append(f"\n[{g}]")
        for k,v in metrics.items():
            if isinstance(v,float):
                lines.append(f"  {k:<30}: {v:.4f}")
    return "\n".join(lines)


def run_evaluation(hypotheses,references,z_pred_list,z_dict_list,
                   pred_adj,true_adj,vad_pred,vad_true):
    results={}
    if references:
        for p in hypotheses:
            if p not in references: continue
            h,r=hypotheses[p],references[p]
            if not h or not r: continue
            results[f"text_{p}"]={**compute_rouge_scores(h,r),
                                   **compute_bleu_scores(h,r)}
    if pred_adj is not None and true_adj is not None:
        results["graph"]=graph_alignment_score(pred_adj,true_adj)
    if z_dict_list:
        acc={}
        for zd in z_dict_list:
            for k,v in perspective_divergence(zd).items():
                acc.setdefault(k,[]).append(v)
        results["perspective"]={k:float(np.mean(v)) for k,v in acc.items()}
    if vad_pred is not None:
        results["emotion"]=emotion_consistency(vad_pred,vad_true)
    if z_pred_list:
        results["latent"]=latent_space_quality(z_pred_list)
    return results


if __name__=="__main__":
    h="The detective solves the mystery with great skill."
    r="A detective uncovers clues and resolves the case."
    print("ROUGE:",compute_rouge_scores([h],[r]))
    print("BLEU:", compute_bleu_scores([h],[r]))
    zd={"protagonist":torch.randn(64),"antagonist":torch.randn(64),"narrator":torch.randn(64)}
    print("Persp div:",perspective_divergence(zd))
    print("Latent quality:",latent_space_quality([torch.randn(128) for _ in range(10)]))
    # Mock ablation results
    mock={"full":{"final_loss":1.23,"graph_f1":0.61,"perspective_divergence":12.4,"history":{},"overrides":{}},
          "no_emotion":{"final_loss":1.87,"graph_f1":0.45,"perspective_divergence":8.1,"history":{},"overrides":{}}}
    print(format_ablation_table(mock))
    print("metrics.py ✓")
