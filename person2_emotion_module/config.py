from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
PROJECT_ROOT = PARENT_DIR
SCENE_INDEX_FILE = PROJECT_ROOT / "data" / "scene_index.csv"
OUTPUT_DIR = BASE_DIR / "output"
AUDIO_CLIPS_DIR = OUTPUT_DIR / "audio_clips"
FEATURES_DIR = OUTPUT_DIR / "features"
SCENE_EMOTIONS_DIR = OUTPUT_DIR / "scene_emotions"
SAMPLE_RATE = 16000
PROJECT_EMOTIONS = ["happy", "sad", "angry", "fearful", "calm", "tense"]
AUDIO_MODEL_NAME = "superb/wav2vec2-base-superb-er"
for folder in [OUTPUT_DIR, AUDIO_CLIPS_DIR, FEATURES_DIR, SCENE_EMOTIONS_DIR]:
    folder.mkdir(parents = True, exist_ok = True)