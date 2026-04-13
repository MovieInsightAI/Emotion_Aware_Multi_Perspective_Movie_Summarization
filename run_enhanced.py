#!/usr/bin/env python3
"""
run_enhanced.py
================
OCP-ADDITIVE helper script — run the enhanced extension layer.
Does NOT modify any original file. Safe to run alongside the original app.

Usage:
    python run_enhanced.py                          # demo with synthetic data
    python run_enhanced.py --report                 # print full research report
    python run_enhanced.py --export results.json    # export JSON results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wrappers.enhanced_pipeline import EnhancedPipeline

DEMO_SCENES = [
    {"happy": 0.55, "sad": 0.15, "angry": 0.10, "fearful": 0.05, "calm": 0.10, "tense": 0.05},
    {"happy": 0.30, "sad": 0.25, "angry": 0.15, "fearful": 0.12, "calm": 0.08, "tense": 0.10},
    {"happy": 0.10, "sad": 0.15, "angry": 0.40, "fearful": 0.18, "calm": 0.05, "tense": 0.12},
    {"happy": 0.08, "sad": 0.12, "angry": 0.35, "fearful": 0.25, "calm": 0.05, "tense": 0.15},
    {"happy": 0.20, "sad": 0.30, "angry": 0.15, "fearful": 0.10, "calm": 0.15, "tense": 0.10},
    {"happy": 0.45, "sad": 0.20, "angry": 0.05, "fearful": 0.05, "calm": 0.20, "tense": 0.05},
]

DEMO_AUDIO = [
    {"happy": 0.6, "sad": 0.1, "angry": 0.05, "fearful": 0.05, "calm": 0.15, "tense": 0.05},
    {"happy": 0.3, "sad": 0.3, "angry": 0.1,  "fearful": 0.1,  "calm": 0.1,  "tense": 0.1},
    {"happy": 0.1, "sad": 0.1, "angry": 0.5,  "fearful": 0.15, "calm": 0.05, "tense": 0.1},
    {"happy": 0.05,"sad": 0.1, "angry": 0.4,  "fearful": 0.3,  "calm": 0.05, "tense": 0.1},
    {"happy": 0.2, "sad": 0.35,"angry": 0.1,  "fearful": 0.1,  "calm": 0.15, "tense": 0.1},
    {"happy": 0.5, "sad": 0.2, "angry": 0.05, "fearful": 0.05, "calm": 0.15, "tense": 0.05},
]

DEMO_SUBTITLE = [
    {"happy": 0.4, "sad": 0.2, "angry": 0.1,  "fearful": 0.05, "calm": 0.2,  "tense": 0.05},
    {"happy": 0.3, "sad": 0.2, "angry": 0.2,  "fearful": 0.15, "calm": 0.05, "tense": 0.1},
    {"happy": 0.1, "sad": 0.2, "angry": 0.3,  "fearful": 0.2,  "calm": 0.05, "tense": 0.15},
    {"happy": 0.1, "sad": 0.15,"angry": 0.3,  "fearful": 0.25, "calm": 0.1,  "tense": 0.1},
    {"happy": 0.2, "sad": 0.25,"angry": 0.2,  "fearful": 0.1,  "calm": 0.15, "tense": 0.1},
    {"happy": 0.4, "sad": 0.2, "angry": 0.05, "fearful": 0.05, "calm": 0.25, "tense": 0.05},
]


def main():
    parser = argparse.ArgumentParser(description="EmotionCine Enhanced Extension Runner")
    parser.add_argument("--report", action="store_true", help="Print full research report")
    parser.add_argument("--export", metavar="FILE", help="Export results to JSON file")
    parser.add_argument("--test-calibration", action="store_true", help="Run calibration diagnostics only")
    parser.add_argument("--metrics", action="store_true", help="Show extended metrics_plus report")
    args = parser.parse_args()

    print("EmotionCine — Enhanced Extension Layer")
    print("OCP-compliant: zero original files modified")
    print("-" * 50)

    if args.test_calibration:
        from calibration.confidence.calibration_layer import EmotionCalibrator, diagnose_calibration
        cal = EmotionCalibrator()
        diagnostics = diagnose_calibration(DEMO_SCENES)
        calibrated = cal.calibrate_all(DEMO_SCENES)
        print(f"\nCalibration diagnostics for {len(diagnostics)} scenes:")
        for d in diagnostics:
            status = "⚠ DEGENERATE" if d.is_degenerate else ("⚠ SPARSE" if d.is_sparse else "✓ OK")
            print(f"  Scene {d.scene_id}: {status} | max={d.max_prob:.3f} | H={d.entropy:.3f}")
        print("\nCalibrated outputs (scene 0):", calibrated[0])
        return

    print("Running enhanced pipeline on demo data...")
    pipeline = EnhancedPipeline()
    result = pipeline.run(
        DEMO_SCENES,
        audio_scores=DEMO_AUDIO,
        subtitle_scores=DEMO_SUBTITLE,
        scene_summaries=[
            "The protagonist starts the journey with optimism.",
            "Tension rises as the threat becomes clear.",
            "Confrontation reaches its peak — anger and fear dominate.",
            "Crisis deepens; the characters face their darkest moment.",
            "A turning point — hope begins to emerge.",
            "Resolution: emotional calm and narrative closure.",
        ],
    )

    if args.metrics:
        from metrics_plus import temporal_consistency, cross_modal_agreement, narrative_coherence
        print("\n── metrics_plus — Extended Metrics ──────────────────")
        tc = temporal_consistency(result.calibrated_emotions)
        print(f"  Temporal consistency:  {tc:.4f}")
        cma = result.metrics.get("cross_modal_agreement")
        if cma is not None:
            print(f"  Cross-modal agreement: {cma:.4f}")
        nc = result.metrics.get("narrative_coherence")
        if nc is not None:
            print(f"  Narrative coherence:   {nc:.4f}")
        print(f"  Causal edges:          {len(result.causal_edges)}")
        print(f"  Perspective conflicts:  {len(result.perspective_conflicts)} scenes scored")
        return

    if args.report:
        print("\n" + result.research_report)
    else:
        print(f"\n✓ Pipeline completed")
        print(f"  Arc type:       {result.arc.dominant_arc}")
        print(f"  Peak scenes:    {result.arc.peak_scenes}")
        print(f"  Causal edges:   {len(result.causal_edges)}")
        print(f"  Mean entropy:   {result.metrics.get('mean_entropy', 0):.4f}")
        print(f"  Calibration:    {result.metrics.get('calibration_health', 'N/A')}")
        print(f"\n  Baseline comparison:")
        for name, bl in result.baseline_comparison.items():
            print(f"    vs {name:<15}: Δentropy={bl['delta'].get('Δentropy', 0):+.4f}")
        print(f"\n  Ablation summary:")
        for name, abl in result.ablation_results.items():
            ent = abl['metrics'].get('mean_entropy', 0)
            print(f"    {name:<25}: entropy={ent:.4f}")

    if args.export:
        import dataclasses
        def safe_dict(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return str(obj)

        export_data = {
            "arc_type": result.arc.dominant_arc,
            "peak_scenes": result.arc.peak_scenes,
            "narrative_tension": result.arc.narrative_tension,
            "metrics": result.metrics,
            "baseline_comparison": result.baseline_comparison,
            "ablation_results": result.ablation_results,
            "calibration_summary": result.calibration_summary,
            "causal_edges": result.causal_edges[:20],   # truncate for readability
            "research_report": result.research_report,
        }
        out_path = Path(args.export)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\n✓ Results exported to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
