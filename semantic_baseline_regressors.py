# semantic_baseline_regressors.py

import os
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Tokenizer, GPT2Model
from scipy.stats import pearsonr

from experiment_subject_list import stories

RESULT_DIR = "results"
DIAG_DIR = os.path.join(RESULT_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

TR = 1.5

def load_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

def get_unique_stories(stories):
    by_name = {}
    for s in stories:
        if s["name"] not in by_name:
            by_name[s["name"]] = s
    return list(by_name.values())

def get_chunks_and_embeddings(story, tokenizer, model, tr=TR, max_length=128):
    with open(story["transcript_file"], "r", encoding="utf-8") as f:
        words = f.read().strip().split()

    events_df = pd.read_csv(story["events_file"], sep="\t")
    story_event = events_df[events_df["trial_type"] == "story"]
    if story_event.empty:
        raise ValueError(f"No 'story' event found for {story['name']}")

    onset = float(story_event.iloc[0]["onset"])
    duration = float(story_event.iloc[0]["duration"])
    story_TRs = int(duration // tr)
    if story_TRs <= 0:
        raise ValueError(f"Non-positive story_TRs for {story['name']}")

    words_per_TR = int(np.floor(len(words) / story_TRs))
    if words_per_TR == 0:
        raise ValueError(f"words_per_TR=0 for {story['name']}")

    word_chunks = [
        " ".join(words[i * words_per_TR : (i + 1) * words_per_TR])
        for i in range(story_TRs)
    ]
    leftover = words[story_TRs * words_per_TR :]
    if leftover:
        word_chunks[-1] += " " + " ".join(leftover)

    # GPT-2 embeddings
    embeddings = []
    for chunk in word_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0).mean(dim=0)
        embeddings.append(emb.numpy())

    X = np.stack(embeddings)  # (T, 768)
    return word_chunks, X

def compute_baseline_regressors(word_chunks):
    word_counts = []
    mean_char_len = []

    for chunk in word_chunks:
        tokens = chunk.split()
        if len(tokens) == 0:
            word_counts.append(0)
            mean_char_len.append(0.0)
            continue
        word_counts.append(len(tokens))
        char_lens = [len(w) for w in tokens]
        mean_char_len.append(float(np.mean(char_lens)))

    return np.array(word_counts), np.array(mean_char_len)

def run_baseline_analysis():
    tokenizer, model = load_models()
    unique_stories = get_unique_stories(stories)

    rows = []

    for story in unique_stories:
        print(f"Computing baseline regressors for story: {story['name']}")

        word_chunks, X = get_chunks_and_embeddings(story, tokenizer, model, tr=TR)

        # PCA on embeddings
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=5)
        scores = pca.fit_transform(X_scaled)
        pc1 = scores[:, 0]

        # Baselines
        word_counts, mean_char_len = compute_baseline_regressors(word_chunks)
        t = np.arange(len(pc1))

        # Correlations
        def safe_pearson(x, y):
            if len(x) < 3:
                return np.nan, np.nan
            return pearsonr(x, y)

        r_wc, p_wc = safe_pearson(pc1, word_counts)
        r_char, p_char = safe_pearson(pc1, mean_char_len)
        r_time, p_time = safe_pearson(pc1, t)

        rows.append({
            "story": story["name"],
            "r_PC1_wordcount": r_wc,
            "p_PC1_wordcount": p_wc,
            "r_PC1_mean_charlen": r_char,
            "p_PC1_mean_charlen": p_char,
            "r_PC1_time": r_time,
            "p_PC1_time": p_time
        })

        # You can also save the time series for plotting if you like:
        df_ts = pd.DataFrame({
            "PC1": pc1,
            "word_count": word_counts,
            "mean_char_len": mean_char_len,
            "time_TR": t
        })
        df_ts.to_csv(
            os.path.join(DIAG_DIR, f"{story['name']}_pc1_and_baselines.csv"),
            index=False
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG_DIR, "pc1_vs_baselines_correlations.csv"), index=False)
    print("Saved baseline regressors correlations in:", DIAG_DIR)


if __name__ == "__main__":
    run_baseline_analysis()
