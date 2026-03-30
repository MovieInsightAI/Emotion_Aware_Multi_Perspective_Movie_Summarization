from transformers import pipeline

PROJECT_LABELS = ["happy", "sad", "angry", "fearful", "calm", "tense"]


class SubtitleEmotionHint:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def predict(self, text: str):
        text = str(text).strip()
        if not text:
            return {label: 0.0 for label in PROJECT_LABELS}

        result = self.classifier(text, candidate_labels=PROJECT_LABELS, multi_label=True)
        scores = dict(zip(result["labels"], result["scores"]))

        final_scores = {label: float(scores.get(label, 0.0)) for label in PROJECT_LABELS}
        return final_scores