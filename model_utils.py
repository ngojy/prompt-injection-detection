import torch
import torch.nn as nn
import numpy as np
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    p = np.array(p, dtype=np.float64) + 1e-10 # small epsilon to avoid log(0)
    q = np.array(q, dtype=np.float64) + 1e-10
    return np.sum(p * np.log(p / q))

# Naive Bayes classifier for text
class NaiveBayesClassifier:
    def __init__(self, use_tfidf=True):
        # TfidfVectorizer weighs words by importance (rare words across docs get higher weight)
        # CountVectorizer just counts raw word occurrences
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        else:
            self.vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        self.classifier = MultinomialNB()
        self.is_fitted = False
    
    def fit(self, texts, labels):
        # Convert raw texts into a numerical feature matrix (sparse matrix of word counts/tfidf scores)
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True
    
    def predict_proba(self, texts):
        # Ensure the model is trained before trying to predict
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        # Transform new texts using the same vocabulary learned during fit
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def predict(self, texts):
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

# Embedding model
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
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

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
    model_nb = None
    model_emb = None
    model_comb = None
    scalers = {}

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

    # combined model (expects entropy + kl + naive_bayes + embedding_dim) = 1 + 1 + 1 + 384 = 387 input features
    try:
        comb_input_dim = 387
        model_comb = EntropyClassifier(input_dim=comb_input_dim).to(device)
        model_comb.load_state_dict(torch.load("combined_model.pth", map_location=device))
        model_comb.eval()
    except Exception:
        model_comb = None

    # naive bayes model
    try:
        with open("naive_bayes_model.pkl", "rb") as f:
            model_nb = pickle.load(f)
    except Exception:
        model_nb = None

    for name in ["entropy", "kl", "nb", "emb"]:
        try:
            with open(f"scaler_{name}.pkl", "rb") as f:
                scalers[name] = pickle.load(f)
        except Exception:
            scalers[name] = None

    return model_entropy, model_kl, model_nb, model_emb, model_comb, scalers


# Prediction
def predict_text(text, feature_extractor, model, mode, scalers=None):
    if model is None:
        return {"label": "N/A", "prob": 0.0}

    try:
        features = feature_extractor.extract_features(text)
    except Exception:
        return {"label": "N/A", "prob": 0.0}

    def scale_scalar(value, key):
        """Scale a single scalar value using the fitted scaler."""
        if scalers and scalers.get(key) is not None:
            return float(scalers[key].transform([[value]])[0][0])
        return float(value)

    def scale_array(arr_2d, key):
        """Scale a 2D array (e.g. 1×384 embedding) using the fitted scaler."""
        if scalers and scalers.get(key) is not None:
            return scalers[key].transform(arr_2d)
        return arr_2d

    try:
        if mode == "entropy":
            val = scale_scalar(features[0], "entropy")
            inp = torch.tensor([[val]], dtype=torch.float32).to(device)

        elif mode == "kl":
            val = scale_scalar(features[1], "kl")
            inp = torch.tensor([[val]], dtype=torch.float32).to(device)

        elif mode == "emb":
            emb_model = load_embedding_model()
            emb = (
                emb_model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
                if emb_model is not None
                else np.zeros(384, dtype=np.float32)
            )
            emb_scaled = scale_array(emb.reshape(1, -1), "emb")  # shape (1, 384)
            inp = torch.tensor(emb_scaled, dtype=torch.float32).to(device)

        elif mode == "comb":
            # Get NB probability
            try:
                with open("naive_bayes_model.pkl", "rb") as f:
                    nb_model = pickle.load(f)
                nb_prob = float(predict_nb_text(text, nb_model).get("prob", 0.0))
            except Exception:
                nb_prob = 0.0

            # Get embedding
            emb_model = load_embedding_model()
            emb = (
                emb_model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
                if emb_model is not None
                else np.zeros(384, dtype=np.float32)
            )

            # Scale each feature independently using its own scaler
            entropy_s = scale_scalar(features[0], "entropy") # scalar
            kl_s      = scale_scalar(features[1], "kl") # scalar
            nb_s      = scale_scalar(nb_prob, "nb") # scalar
            emb_s     = scale_array(emb.reshape(1, -1), "emb") # shape (1, 384)

            # Concatenate in same order as training:
            # [entropy(1), kl(1), nb(1), emb(384)] = 387
            arr = np.hstack([
                [[entropy_s]],  # (1, 1)
                [[kl_s]],       # (1, 1)
                [[nb_s]],       # (1, 1)
                emb_s           # (1, 384)
            ]).astype(np.float32)
            inp = torch.tensor(arr, dtype=torch.float32).to(device)

        else:
            return {"label": "N/A", "prob": 0.0}

    except Exception:
        return {"label": "N/A", "prob": 0.0}

    try:
        model.eval()
        with torch.no_grad():
            out = model(inp).cpu().numpy().flatten()
            prob = float(torch.sigmoid(torch.tensor(out[0], dtype=torch.float32)).item()) if out.size > 0 else 0.0
            label = "Malicious" if prob > 0.5 else "Benign"
            return {"label": label, "prob": prob}
    except Exception:
        return {"label": "N/A", "prob": 0.0}

    try:
        model.eval()
        with torch.no_grad():
            out = model(inp).cpu().numpy().flatten()
            prob = float(torch.sigmoid(torch.tensor(out[0], dtype=torch.float32)).item()) if out.size > 0 else 0.0
            label = "Malicious" if prob > 0.5 else "Benign"
            return {"label": label, "prob": prob}
    except Exception:
        return {"label": "N/A", "prob": 0.0}

def predict_nb_text(text, model):
    """
    Predict using naive Bayes model
    """
    if model is None:
        return {"label": "N/A", "prob": 0.0}
    
    try:
        probs = model.predict_proba([text])
        prob = float(probs[0][1])  # probability of class 1 (malicious)
        label = "Malicious" if prob > 0.5 else "Benign"
        return {"label": label, "prob": prob}
    except Exception:
        return {"label": "N/A", "prob": 0.0}