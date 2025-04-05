from flask import Flask, render_template, request
try:
    import pandas as pd
except ImportError:
    raise ImportError("The pandas library is required but not installed. Please install it using 'pip install pandas'.")
import re
from collections import defaultdict
import math
import nltk
try:
    from nltk.corpus import stopwords
except ImportError:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load processed metadata
df = pd.read_csv("pexels_MoS2_final_processed.csv")
image_lookup = dict(zip(df["id"].astype(str), df["image_url"]))

# Load NLP tools
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = text.split()
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

# Build inverted index & stats
inverted_index = defaultdict(set)
term_frequencies = defaultdict(lambda: defaultdict(int))
doc_lengths = {}

for _, row in df.iterrows():
    doc_id = str(row["id"])
    tokens = str(row["processed_text"]).split()
    doc_lengths[doc_id] = len(tokens)
    for token in tokens:
        inverted_index[token].add(doc_id)
        term_frequencies[token][doc_id] += 1

N = len(doc_lengths)
avgdl = sum(doc_lengths.values()) / N
collection_length = sum(doc_lengths.values())
collection_freqs = {term: sum(doc_tf.values()) for term, doc_tf in term_frequencies.items()}

# VSM
def vsm(query_tokens):
    scores = defaultdict(float)
    for term in query_tokens:
        df = len(term_frequencies.get(term, {}))
        idf = math.log((N + 1) / (df + 1))
        for doc_id, tf in term_frequencies[term].items():
            scores[doc_id] += tf * idf
    for doc_id in scores:
        scores[doc_id] /= doc_lengths[doc_id]
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# BM25
def bm25(query_tokens):
    k1, b = 1.5, 0.75
    scores = defaultdict(float)
    for term in query_tokens:
        df = len(term_frequencies.get(term, {}))
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for doc_id, tf in term_frequencies[term].items():
            dl = doc_lengths[doc_id]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avgdl))
            scores[doc_id] += idf * (numerator / denominator)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Language Model
def language_model(query_tokens):
    lambda_param = 0.1
    scores = {}
    for doc_id in doc_lengths:
        doc_len = doc_lengths[doc_id]
        score = 0
        for term in query_tokens:
            tf_d = term_frequencies.get(term, {}).get(doc_id, 0)
            tf_c = collection_freqs.get(term, 0)
            prob_d = tf_d / doc_len if doc_len else 0
            prob_c = tf_c / collection_length if collection_length else 0
            smoothed = (1 - lambda_param) * prob_d + lambda_param * prob_c
            if smoothed > 0:
                score += math.log(smoothed)
        scores[doc_id] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Web Route
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    model_used = ""

    if request.method == "POST":
        query = request.form["query"]
        model = request.form["model"]
        tokens = preprocess_text(query)

        # Retrieve using selected model
        if model == "vsm":
            ranked = vsm(tokens)
            model_used = "VSM"
        elif model == "bm25":
            ranked = bm25(tokens)
            model_used = "BM25"
        else:
            ranked = language_model(tokens)
            model_used = "Language Model"

        # Format results
        for doc_id, score in ranked[:10]:
            row = df[df["id"] == int(doc_id)].iloc[0]

            # Highlight query terms in description
            raw_desc = str(row.get("description", ""))
            highlighted = raw_desc
            for token in tokens:
                highlighted = re.sub(f"(?i)({token})", r"<span class='highlight'>\1</span>", highlighted)

            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "image_url": row["image_url"],
                "photographer": row.get("photographer", "N/A"),
                "description": highlighted
            })

    return render_template("index.html", results=results, model=model_used)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
