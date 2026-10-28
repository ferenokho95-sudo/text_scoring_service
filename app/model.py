from pathlib import Path
import hashlib
import json
import joblib
import numpy as np


FEATURES = ("characters", "words", "cyrillic_ratio", "oov_ratio", "positive_score")


class Scorer:
    def __init__(self, path: Path):
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model_path = path / "model.joblib"
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != self.metadata["model_sha256"]:
            raise ValueError("Model checksum mismatch. Re-run training.")
        # Only load artifacts created by our training command, never user uploads.
        self.pipeline = joblib.load(model_path)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.tokenize = self.vectorizer.build_tokenizer()
        self.preprocess = self.vectorizer.build_preprocessor()
        self.positive_index = list(self.pipeline.classes_).index("positive")
        self.reference = np.load(path / "reference.npy", allow_pickle=False)
        if self.reference.ndim != 2 or self.reference.shape[1] != len(FEATURES):
            raise ValueError("Invalid reference features")

    @property
    def version(self):
        return self.metadata["model_version"]

    def predict(self, texts: list[str]):
        scores = self.pipeline.predict_proba(texts)[:, self.positive_index]
        rows = []
        for text, score in zip(texts, scores):
            tokens = self.tokenize(self.preprocess(text))
            letters = sum(c.isalpha() for c in text)
            cyrillic = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in text)
            oov = sum(token not in self.vectorizer.vocabulary_ for token in tokens)
            rows.append([len(text), len(tokens), cyrillic / max(letters, 1),
                         oov / max(len(tokens), 1), float(score)])
        return scores, np.asarray(rows, dtype=float)
