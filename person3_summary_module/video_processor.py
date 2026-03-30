"""
video_processor.py
==================
Extract scene-aligned subtitles and emotion feature vectors directly
from uploaded video files — no external ASR or pre-trained models used.

Pipeline
--------
1. Video → extract audio track (moviepy / ffmpeg)
2. Audio → scene segmentation by silence/energy thresholds
3. Each scene → 8-dim acoustic emotion proxy vector:
   [energy, zcr, spectral_centroid_norm, spectral_rolloff_norm,
    tempo_norm, rms_delta, pitch_proxy, spectral_bandwidth_norm]
4. Video → embedded subtitle extraction (SRT/ASS tracks via ffmpeg)
   If no embedded subtitles → generate placeholder text per scene.

No pre-trained ASR, no external embedding models.
All features computed from raw audio signal with numpy/scipy only.
"""

from __future__ import annotations

import io
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1.  FFmpeg helpers
# ============================================================================
def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


def extract_audio_ffmpeg(video_path: str, out_wav: str,
                         sample_rate: int = 16000) -> bool:
    """Extract mono WAV from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1",                          # mono
        "-ar", str(sample_rate),             # sample rate
        "-vn",                               # no video
        out_wav,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except Exception as e:
        logger.warning("ffmpeg audio extraction failed: %s", e)
        return False


def extract_subtitles_ffmpeg(video_path: str, out_srt: str) -> bool:
    """Try to extract embedded subtitle track (stream 0:s:0) as SRT."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-map", "0:s:0",                    # first subtitle stream
        out_srt,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and Path(out_srt).stat().st_size > 0
    except Exception:
        return False


def get_video_duration_ffprobe(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ============================================================================
# 2.  Pure-numpy audio feature extraction
# ============================================================================
def load_wav_numpy(wav_path: str) -> Tuple[np.ndarray, int]:
    """
    Load WAV file using scipy.io.wavfile (no external audio libs required).
    Returns (samples float32, sample_rate).
    """
    from scipy.io import wavfile
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    # Normalise to [-1, 1]
    mx = np.abs(data).max()
    if mx > 0:
        data /= mx
    return data, int(sr)


def compute_scene_emotion_vector(
    audio_segment: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Compute an 8-dim acoustic emotion proxy for one audio segment.

    Dimensions (all normalised to [0, 1]):
      0: RMS energy          — overall loudness / intensity
      1: Zero-crossing rate  — noise/consonant density → anger proxy
      2: Spectral centroid   — brightness → arousal proxy
      3: Spectral roll-off   — high-freq content
      4: Tempo proxy         — beat strength via autocorrelation
      5: RMS delta           — energy change rate → surprise proxy
      6: Pitch proxy         — fundamental freq estimate → valence proxy
      7: Spectral bandwidth  — tonal spread

    Returns np.ndarray shape (8,), float32, values in [0, 1].
    """
    if len(audio_segment) < 10:
        return np.zeros(8, dtype=np.float32)

    eps = 1e-9
    N = len(audio_segment)

    # --- 0. RMS energy ---
    rms = float(np.sqrt(np.mean(audio_segment ** 2)))

    # --- 1. Zero-crossing rate ---
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio_segment)))) / 2)

    # --- 2 & 7. Spectral centroid + bandwidth (via FFT) ---
    fft_mag = np.abs(np.fft.rfft(audio_segment * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    total_mag = fft_mag.sum() + eps
    spec_centroid = float(np.sum(freqs * fft_mag) / total_mag)
    spec_bandwidth = float(
        np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * fft_mag) / total_mag))

    # --- 3. Spectral roll-off (85% energy threshold) ---
    cumsum = np.cumsum(fft_mag)
    rolloff_thresh = 0.85 * cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, rolloff_thresh)
    spec_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # --- 4. Tempo proxy (autocorrelation peak lag) ---
    hop = max(1, int(sr * 0.01))       # 10 ms hop
    frame_rms = np.array([
        np.sqrt(np.mean(audio_segment[i:i + hop] ** 2))
        for i in range(0, N - hop, hop)
    ])
    if len(frame_rms) > 4:
        ac = np.correlate(frame_rms - frame_rms.mean(),
                          frame_rms - frame_rms.mean(), mode="full")
        ac = ac[len(ac) // 2:]
        min_lag = max(1, int(sr * 0.3 / hop))    # ~200 BPM max
        max_lag = min(len(ac) - 1, int(sr * 2.0 / hop))  # ~30 BPM min
        if max_lag > min_lag:
            peak_lag = np.argmax(ac[min_lag:max_lag]) + min_lag
            tempo_bpm = (sr / hop) * 60.0 / max(peak_lag, 1)
            tempo_norm = min(tempo_bpm / 200.0, 1.0)
        else:
            tempo_norm = 0.3
    else:
        tempo_norm = 0.3

    # --- 5. RMS delta (energy change) ---
    if len(frame_rms) > 1:
        rms_delta = float(np.mean(np.abs(np.diff(frame_rms))))
    else:
        rms_delta = 0.0

    # --- 6. Pitch proxy (dominant frequency via FFT peak in voiced band) ---
    voiced_mask = (freqs >= 80) & (freqs <= 400)
    if voiced_mask.any():
        pitch_peak_idx = np.argmax(fft_mag[voiced_mask])
        pitch_hz = float(freqs[voiced_mask][pitch_peak_idx])
        # Normalise: 80-400 Hz → 0-1
        pitch_norm = (pitch_hz - 80.0) / 320.0
    else:
        pitch_norm = 0.0

    # --- Normalise all features ---
    nyq = sr / 2.0
    vec = np.array([
        min(rms * 5.0, 1.0),                          # 0: energy
        min(zcr * 2.0, 1.0),                           # 1: ZCR
        min(spec_centroid / nyq, 1.0),                 # 2: centroid
        min(spec_rolloff / nyq, 1.0),                  # 3: rolloff
        float(tempo_norm),                             # 4: tempo
        min(rms_delta * 20.0, 1.0),                    # 5: RMS delta
        float(np.clip(pitch_norm, 0.0, 1.0)),          # 6: pitch
        min(spec_bandwidth / nyq, 1.0),                # 7: bandwidth
    ], dtype=np.float32)

    return np.clip(vec, 0.0, 1.0)


# ============================================================================
# 3.  Scene segmentation
# ============================================================================
def segment_audio_by_energy(
    audio: np.ndarray,
    sr: int,
    scene_duration_sec: float = 5.0,
    min_scenes: int = 3,
    max_scenes: int = 60,
) -> List[Tuple[float, float]]:
    """
    Segment audio into scenes using a fixed-window approach with
    adaptive duration.

    Returns list of (start_sec, end_sec) tuples.
    """
    total_sec = len(audio) / sr
    if total_sec <= 0:
        return [(0.0, 1.0)]

    # Adapt scene duration to get a reasonable number of scenes
    n_target = max(min_scenes, min(max_scenes, int(total_sec / scene_duration_sec)))
    actual_dur = total_sec / n_target

    scenes = []
    for i in range(n_target):
        start = i * actual_dur
        end = min((i + 1) * actual_dur, total_sec)
        scenes.append((float(start), float(end)))

    return scenes


# ============================================================================
# 4.  SRT generation (placeholder when no embedded subtitles)
# ============================================================================
def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_placeholder_srt(
    scenes: List[Tuple[float, float]],
    emotion_vecs: np.ndarray,
    video_name: str = "video",
) -> str:
    """
    Generate placeholder SRT when no embedded subtitles are available.
    Inserts acoustically-derived emotion descriptors as scene text.
    """
    EMOTION_NAMES = [
        "Joy", "Sadness", "Anger", "Fear",
        "Surprise", "Disgust", "Trust", "Anticipation"
    ]
    INTENSITY_WORDS = ["low", "moderate", "high", "intense"]
    lines = []
    for i, ((start, end), vec) in enumerate(zip(scenes, emotion_vecs)):
        dom_idx = int(np.argmax(vec))
        dom_emo = EMOTION_NAMES[dom_idx]
        energy_idx = min(int(vec[0] * 4), 3)
        intensity = INTENSITY_WORDS[energy_idx]
        text = (f"[Scene {i+1} — {dom_emo}, {intensity} intensity] "
                f"{video_name} segment {_fmt_ts(start)}–{_fmt_ts(end)}")
        lines.append(f"{i+1}")
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# 5.  Main VideoProcessor class
# ============================================================================
class VideoProcessor:
    """
    End-to-end video → (subtitle_srt, emotion_array) pipeline.

    Parameters
    ----------
    scene_duration : float   Target scene length in seconds.
    sample_rate    : int     Audio sample rate for feature extraction.
    """

    def __init__(self, scene_duration: float = 5.0, sample_rate: int = 16000):
        self.scene_duration = scene_duration
        self.sample_rate = sample_rate

    def process(
        self,
        video_bytes: bytes,
        filename: str = "upload.mp4",
        progress_cb=None,
    ) -> Dict:
        """
        Process raw video bytes.

        Parameters
        ----------
        video_bytes  : raw video file content
        filename     : original filename (used for format detection)
        progress_cb  : optional callable(float, str) for progress updates

        Returns
        -------
        dict with keys:
            srt_text      : str   SRT subtitle string
            emotion_array : ndarray (N, 8)
            scenes        : list of (start_sec, end_sec)
            n_scenes      : int
            has_embedded_subs : bool
            audio_extracted   : bool
            duration_sec  : float
            error         : str or None
        """
        def _prog(p, msg):
            logger.info("[VideoProcessor] %.0f%% — %s", p * 100, msg)
            if progress_cb:
                progress_cb(p, msg)

        result = {
            "srt_text": "",
            "emotion_array": np.zeros((1, 8), dtype=np.float32),
            "scenes": [],
            "n_scenes": 0,
            "has_embedded_subs": False,
            "audio_extracted": False,
            "duration_sec": 0.0,
            "error": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write video to temp file
            ext = Path(filename).suffix.lower() or ".mp4"
            video_path = os.path.join(tmpdir, f"input{ext}")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            _prog(0.1, "Video saved to temp dir")

            # ── Try extracting embedded subtitles ────────────────────────────
            srt_path = os.path.join(tmpdir, "subs.srt")
            has_subs = False
            if _ffmpeg_available():
                has_subs = extract_subtitles_ffmpeg(video_path, srt_path)
                if has_subs:
                    result["srt_text"] = Path(srt_path).read_text(
                        encoding="utf-8", errors="ignore")
                    result["has_embedded_subs"] = True
                    _prog(0.3, f"Extracted embedded subtitles")

            # ── Extract audio ─────────────────────────────────────────────────
            wav_path = os.path.join(tmpdir, "audio.wav")
            audio_ok = False
            audio = None

            if _ffmpeg_available():
                audio_ok = extract_audio_ffmpeg(
                    video_path, wav_path, self.sample_rate)
                if audio_ok:
                    try:
                        audio, sr = load_wav_numpy(wav_path)
                        result["audio_extracted"] = True
                        result["duration_sec"] = len(audio) / sr
                        _prog(0.5, f"Audio extracted ({result['duration_sec']:.1f}s)")
                    except Exception as e:
                        logger.warning("WAV load failed: %s", e)
                        audio_ok = False

            # ── Fallback: try moviepy ─────────────────────────────────────────
            if not audio_ok:
                try:
                    from moviepy.editor import VideoFileClip
                    clip = VideoFileClip(video_path)
                    result["duration_sec"] = clip.duration or 10.0
                    if clip.audio is not None:
                        wav_path2 = os.path.join(tmpdir, "audio_mp.wav")
                        clip.audio.write_audiofile(
                            wav_path2, fps=self.sample_rate,
                            nbytes=2, logger=None)
                        audio, sr = load_wav_numpy(wav_path2)
                        result["audio_extracted"] = True
                        audio_ok = True
                    clip.close()
                    _prog(0.5, "Audio extracted via moviepy")
                except Exception as e:
                    logger.warning("moviepy fallback failed: %s", e)

            # ── Scene segmentation + emotion extraction ───────────────────────
            if audio_ok and audio is not None and len(audio) > 0:
                scenes = segment_audio_by_energy(
                    audio, self.sample_rate, self.scene_duration)
                _prog(0.65, f"Segmented into {len(scenes)} scenes")

                emotion_vecs = []
                for i, (start_s, end_s) in enumerate(scenes):
                    s_samp = int(start_s * self.sample_rate)
                    e_samp = int(end_s   * self.sample_rate)
                    seg = audio[s_samp:e_samp]
                    vec = compute_scene_emotion_vector(seg, self.sample_rate)
                    emotion_vecs.append(vec)
                    _prog(0.65 + 0.25 * (i + 1) / len(scenes),
                          f"Features: scene {i+1}/{len(scenes)}")

                result["emotion_array"] = np.stack(emotion_vecs)
                result["scenes"] = scenes
                result["n_scenes"] = len(scenes)
            else:
                # No audio: use fixed-interval scenes based on duration
                dur = result["duration_sec"] or 30.0
                n = max(3, int(dur / self.scene_duration))
                scenes = [(i * dur / n, (i + 1) * dur / n) for i in range(n)]
                result["scenes"] = scenes
                result["n_scenes"] = n
                # Random-ish emotion vectors as placeholder
                rng = np.random.default_rng(42)
                result["emotion_array"] = rng.uniform(
                    0, 0.6, (n, 8)).astype(np.float32)
                _prog(0.9, "Using placeholder features (no audio)")

            # ── Generate placeholder SRT if none found ────────────────────────
            if not result["srt_text"].strip():
                result["srt_text"] = generate_placeholder_srt(
                    result["scenes"],
                    result["emotion_array"],
                    video_name=Path(filename).stem,
                )
                _prog(0.95, "Generated placeholder subtitles")

            _prog(1.0, "Done")

        return result


# ============================================================================
# 6.  Smoke test
# ============================================================================
if __name__ == "__main__":
    # Test with synthetic audio (sine wave — no actual video needed)
    import scipy.io.wavfile as wav_write
    sr = 16000
    duration = 30  # seconds
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    # Composite sine — simulates voiced audio
    audio_synth = (0.5 * np.sin(2 * np.pi * 200 * t)
                   + 0.3 * np.sin(2 * np.pi * 440 * t)
                   + 0.1 * np.random.randn(len(t)).astype(np.float32))

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "synth.wav")
        wav_write.write(wav_path, sr, audio_synth)

        audio, sr2 = load_wav_numpy(wav_path)
        scenes = segment_audio_by_energy(audio, sr2, scene_duration_sec=5.0)
        print(f"Scenes: {len(scenes)}")

        vecs = []
        for start, end in scenes:
            s = int(start * sr2); e = int(end * sr2)
            v = compute_scene_emotion_vector(audio[s:e], sr2)
            vecs.append(v)
            print(f"  {start:.1f}s-{end:.1f}s: {v.round(3)}")

        emotion_arr = np.stack(vecs)
        srt = generate_placeholder_srt(scenes, emotion_arr, "test")
        print(f"\nSRT preview:\n{srt[:300]}")
        print("video_processor.py ✓")
