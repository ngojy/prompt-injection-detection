import torch
import torch.nn as nn
import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pickle

device = torch.device("cpu")

# Shannon entropy calculation
def shannon_entropy(text):
    if len(text) == 0:
        return 0.0

    char_counts = Counter(text)
    total_chars = len(text)

    probabilities = [count / total_chars for count in char_counts.values()]

    entropy = -sum(p * math.log2(p) for p in probabilities)

    return entropy

# KL divergence calculation
def kl_divergence(p, q):
    p = np.array(p, dtype=np.float64) + 1e-10
    q = np.array(q, dtype=np.float64) + 1e-10
    return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

# Embedding model loading
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_emb_model = None

def load_embedding_model(name="all-MiniLM-L6-v2"):
    global _emb_model
    if SentenceTransformer is None:
        return None
    if _emb_model is None:
        _emb_model = SentenceTransformer(name)
    return _emb_model

# Feature extraction
class EntropyKLFeatureExtractor:
    def __init__(self, benign_texts):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(benign_texts)

        benign_counts = self.vectorizer.transform(benign_texts).toarray()
        self.benign_distribution = np.mean(benign_counts, axis=0) / (np.sum(benign_counts) + 1e-10)

    def extract_features(self, text):
        global_entropy = shannon_entropy(text)

        vec = self.vectorizer.transform([text]).toarray()[0]
        dist = vec / (np.sum(vec) + 1e-10)
        kl_div = kl_divergence(dist, self.benign_distribution)

        specical_ratio = sum(not c.isalnum() for c in text) / (len(text) + 1e-10)

        return np.array([
            global_entropy,
            kl_div,
            specical_ratio,
            len(text)],
            dtype=np.float32)

# Neural network model
class EntropyClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EntropyClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Load Feature Extractor
def load_feature_extractor(path="feature_extractor.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Provide a safe fallback extractor so the GUI and prediction
        # code can still run when the serialized feature extractor
        # isn't present.
        class _FallbackExtractor:
            def extract_features(self, text):
                e = shannon_entropy(text)
                # KL requires a benign distribution; return 0.0 as a
                # conservative default for missing artifact.
                kl = 0.0
                specical_ratio = sum(not c.isalnum() for c in text) / (len(text) + 1e-10)
                return np.array([e, kl, specical_ratio, len(text)], dtype=np.float32)

        return _FallbackExtractor()

def load_models():
    model_entropy = None
    model_kl = None
    model_emb = None
    model_comb = None

    # entropy & kl (single-dim inputs)
    try:
        model_entropy = EntropyClassifier(input_dim=1).to(device)
        model_entropy.load_state_dict(torch.load("entropy_model.pth", map_location=device))
        model_entropy.eval()
    except Exception:
        model_entropy = None

    try:
        model_kl = EntropyClassifier(input_dim=1).to(device)
        model_kl.load_state_dict(torch.load("kl_model.pth", map_location=device))
        model_kl.eval()
    except Exception:
        model_kl = None

    # embedding model
    try:
        emb_input_dim = 384
        model_emb = EntropyClassifier(input_dim=emb_input_dim).to(device)
        model_emb.load_state_dict(torch.load("emb_model.pth", map_location=device))
        model_emb.eval()
    except Exception:
        model_emb = None

    # combined model (expects entropy + kl + embedding_dim)
    try:
        comb_input_dim = 386  # 2 + 384
        model_comb = EntropyClassifier(input_dim=comb_input_dim).to(device)
        model_comb.load_state_dict(torch.load("combined_model.pth", map_location=device))
        model_comb.eval()
    except Exception:
        model_comb = None

    return model_entropy, model_kl, model_emb, model_comb


# Prediction
def predict_text(text, feature_extractor, model, mode="entropy"):
    """
    mode: "entropy", "kl", "emb", or "comb"
    If embedding encoder is available it will be used; otherwise a zero-vector fallback is used.
    """
    if model is None:
        return {"label": "N/A", "prob": 0.0}

    # compute base features
    try:
        features = feature_extractor.extract_features(text)  # [entropy, kl, special_ratio, length]
    except Exception:
        return {"label": "N/A", "prob": 0.0}

    # build input tensor
    try:
        if mode == "entropy":
            inp = torch.tensor([[features[0]]], dtype=torch.float32).to(device)
        elif mode == "kl":
            inp = torch.tensor([[features[1]]], dtype=torch.float32).to(device)
        elif mode in ("emb", "comb"):
            # determine expected embedding dims from model if possible
            total_in = None
            try:
                total_in = model.model[0].weight.shape[1]
            except Exception:
                total_in = None

            if mode == "emb":
                expected_emb_dim = total_in if total_in is not None else 384
            else:  # comb
                expected_emb_dim = (total_in - 2) if (total_in is not None and total_in > 2) else 0

            # prefer runtime embedding model if available
            emb_model = load_embedding_model()
            if emb_model is not None:
                emb = emb_model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
            else:
                emb = np.zeros(expected_emb_dim, dtype=np.float32)

            # if runtime embedding size doesn't match expected, pad/truncate
            if expected_emb_dim is not None and expected_emb_dim > 0:
                if emb.shape[0] != expected_emb_dim:
                    if emb.shape[0] > expected_emb_dim:
                        emb = emb[:expected_emb_dim]
                    else:
                        emb = np.pad(emb, (0, expected_emb_dim - emb.shape[0]), mode="constant", constant_values=0.0)

            if mode == "emb":
                inp = torch.tensor(emb.reshape(1, -1), dtype=torch.float32).to(device)
            else:  # comb: [entropy, kl, emb...]
                arr = np.concatenate([[features[0]], [features[1]], emb])
                inp = torch.tensor(arr.reshape(1, -1), dtype=torch.float32).to(device)
        else:
            return {"label": "N/A", "prob": 0.0}
    except Exception:
        return {"label": "N/A", "prob": 0.0}

    # run model
    try:
        model.eval()
        with torch.no_grad():
            out = model(inp).cpu().numpy().flatten()
            prob = float(out[0]) if out.size > 0 else 0.0
            label = "Malicious" if prob > 0.5 else "Benign"
            return {"label": label, "prob": prob}
    except Exception:
        return {"label": "N/A", "prob": 0.0}