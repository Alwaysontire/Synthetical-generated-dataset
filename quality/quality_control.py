import random, faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
from sacrebleu.metrics import BLEU

bleu = BLEU(effective_order=True)

def Self_BLEU(texts, sample_size=300, refs_per_hyp=50, seed=42) -> float:
    rnd = random.Random(seed)
    texts = list(texts)
    hyps = texts if len(texts) <= sample_size else rnd.sample(texts, sample_size)
    scores = []
    for h in hyps:
        others = [t for t in texts if t != h]
        if not others:
            continue
        refs = others if len(others) <= refs_per_hyp else rnd.sample(others, refs_per_hyp)
        score = bleu.sentence_score(h, refs).score
        scores.append(score)
    return float(sum(scores) / len(scores)) if scores else None







def Near_duplicate_rate(embeddings, threshold=0.95) -> float:
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    D, I = index.search(embeddings, 2)
    nearest_sim = D[:, 1]
    nearest_id = I[:, 1]
    semantic_nearest_duplicate_rate = (nearest_sim >= threshold).mean()
    return semantic_nearest_duplicate_rate


def Class_balance():
    pass

def Semantic_diversity(embeddings, index) -> float:
    vectors = embeddings[np.array(list(index))]
    index = np.asarray(list(index), dtype=np.int64)
    if index.size == 0:
        return np.nan, np.nan
    if index.size == 1:
        return 0.0, 0.0
    centroid = np.mean(vectors, axis=0, keepdims=True)
    centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)
    sims = (vectors @ centroid.T).reshape(-1)
    dists = 1 - sims
    return float(np.median(dists)), float(np.mean(dists))
    


def main():
    dataframe = pd.read_csv("samples/new_data.csv")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    embeddings = model.encode(dataframe["phrase"].tolist(), normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    
    #BLEU

    self_bleu_intent = {}
    for intent, g in dataframe.groupby("intent"):
        score = Self_BLEU(g["phrase"], sample_size=300, refs_per_hyp=50)
        self_bleu_intent[intent] = {"self_bleu" : score, "n": len(g)}

    mean_self_bleu = 0
    self_bleu_n = 0
    for item, key in self_bleu_intent.items():
        # print(f"INTENT - {item}, self_bleu_score - {key['self_bleu']}, amount - {key['n']}")
        mean_self_bleu += key["self_bleu"]
        self_bleu_n += 1
    # sorted(self_bleu_intent.items(), key=lambda x: x[1]["self_bleu"])[:10]
    

    #Near_duplicates
    duplicate_rate = Near_duplicate_rate(embeddings, threshold=0.95)
    print(f"DUPLICATE_RATE {duplicate_rate}\n")

    #Semantic_diversity
    dataframe["embedding_idx"] = range(len(dataframe))
    by_intent = {}
    for intent, g in dataframe.groupby("intent"):
        median, mean = Semantic_diversity(embeddings, g["embedding_idx"])
        by_intent[intent] = {"median_dist": median, "mean_dist": mean, "n": len(g)}
    # sorted(by_intent.items(), key=lambda x: x[1]["median_dist"])[:10]
    mean_semantic = 0
    semantic_n = 0
    for item, key in by_intent.items():
        # print(f"INTENT - {item}, median - {key['median_dist']}, mean - {key['mean_dist']}, amount - {key['n']}")
        mean_semantic += key["mean_dist"]
        semantic_n += 1
    

    duplicate_rate = Near_duplicate_rate(embeddings, threshold=0.95)
    print(f"DUPLICATE_RATE {duplicate_rate}")
    print(f"Mean Semantic_diversity {mean_semantic / semantic_n}")
    print(f"Mean Self_Bleu {mean_self_bleu / self_bleu_n}")




if __name__ == "__main__":
    main()
