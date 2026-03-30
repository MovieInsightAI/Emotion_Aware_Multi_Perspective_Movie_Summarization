from audio_extract import extract_all_scene_audio
from preprocess_audio import preprocess_all_audio
from infer_scene_emotions import infer_scene_emotions


def main():
    print("\nStep 1: Extracting audio clips...")
    extract_all_scene_audio()

    print("\nStep 2: Preprocessing audio...")
    preprocess_all_audio()

    print("\nStep 3: Inferring scene emotions...")
    infer_scene_emotions()

    print("\nDone. Person 2 pipeline completed successfully.")


if __name__ == "__main__":
    main()