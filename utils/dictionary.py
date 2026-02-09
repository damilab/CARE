import os
import json
import argparse
import torch
import clip
from PIL import Image
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
import re
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# ------------------ Utils ------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_token(tok: str):
    return tok.replace("</w>", "")

def keyify(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(" ", "")          
    s = re.sub(r"[^a-z0-9_-]", "", s)  
    return s

# ------------------ Main ------------------
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.vocab_cache), exist_ok=True)

    # ------------------ Load CLIP ------------------
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    vocab_tokens = list(tokenizer.encoder.keys())
    V = len(vocab_tokens)

    # ------------------ Build/load vocab embeddings ------------------
    def build_or_load_vocab_embeddings():
        if os.path.exists(args.vocab_cache):
            return np.load(args.vocab_cache, mmap_mode="r")

        all_feats = []
        with torch.no_grad():
            for s in tqdm(range(0, V, args.batch_size_txt),
                          desc="Encoding vocab (batched)"):
                batch_tokens = vocab_tokens[s:s + args.batch_size_txt]
                toks = clip.tokenize(batch_tokens).to(device)
                text_feat = model.encode_text(toks)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                all_feats.append(text_feat.half().cpu().numpy())
                del toks, text_feat
                torch.cuda.empty_cache()

        arr = np.concatenate(all_feats, axis=0)
        np.save(args.vocab_cache, arr.astype(np.float16))
        return np.load(args.vocab_cache, mmap_mode="r")

    text_features_np = build_or_load_vocab_embeddings()
    D = text_features_np.shape[1]

    def encode_words(words):
        with torch.no_grad():
            toks = clip.tokenize(words).to(device)
            feats = model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float()

    # ------------------ Chunked top-k ------------------
    def topk_sim_over_chunks(img_feat_cpu: np.ndarray, k: int):
        import heapq
        heap = []
        for s in range(0, V, args.chunk_size_sim):
            e = min(s + args.chunk_size_sim, V)
            tf = text_features_np[s:e]
            sims = (tf.astype(np.float32) @ img_feat_cpu.astype(np.float32))
            if len(heap) < k:
                for i, val in enumerate(sims):
                    heapq.heappush(heap, (float(val), s + i))
                    if len(heap) > k:
                        heapq.heappop(heap)
            else:
                th = heap[0][0]
                for i, val in enumerate(sims):
                    fv = float(val)
                    if fv > th:
                        heapq.heapreplace(heap, (fv, s + i))
                        th = heap[0][0]
        heap.sort(reverse=True)
        idxs_sorted = [i for _, i in heap]
        return idxs_sorted

    # ======================== 1) Co-occurring tokens ========================
    most_common_counter = Counter()
    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(args.image_dir)),
                          desc="Processing images"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img = Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
            image = preprocess(img).unsqueeze(0).to(device)

            img_feat = model.encode_image(image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_feat_cpu = img_feat.squeeze(0).cpu().numpy()

            topk_indices = topk_sim_over_chunks(
                img_feat_cpu, args.top_k_per_image
            )
            most_common_counter.update(
                [vocab_tokens[i] for i in topk_indices]
            )

            del image, img_feat
            torch.cuda.empty_cache()

    most_common_tokens = [
        tok for tok, _ in most_common_counter.most_common(args.top_n_tokens)
    ]

    # ======================== 2) Global clustering ========================
    W = encode_words(most_common_tokens)
    C = encode_words([args.reference_word]).T

    similarities = (W @ C).squeeze(1)
    PC = (C @ C.T) / (C.T @ C + 1e-12)
    I = torch.eye(D)
    residuals_all = W @ (I - PC).T
    residual_norms_all = residuals_all.norm(dim=1)

    tsne = TSNE(
        n_components=2,
        random_state=args.seed,
        init="pca",
        learning_rate="auto"
    )
    emb_2d = tsne.fit_transform(W.numpy())

    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        n_init="auto"
    )
    labels = kmeans.fit_predict(emb_2d)

    token_counts = torch.tensor(
        [most_common_counter[t] for t in most_common_tokens],
        dtype=torch.float32
    )

    df = pd.DataFrame({
        "token": most_common_tokens,
        "cluster": labels,
        "count": token_counts.numpy().astype(int),
        "sim": similarities.numpy(),
        "residual_norm": residual_norms_all.numpy(),
    })

    cluster_stats = (
        df.groupby("cluster")
          .agg(
              n_tokens=("token", "size"),
              mean_residual=("residual_norm", "mean")
          )
          .reset_index()
          .sort_values("mean_residual")
    )

    ordered = cluster_stats["cluster"].tolist()
    k_eff = min(args.k_each_side, len(ordered) // 2)
    remove_clusters = set(ordered[:k_eff] + ordered[-k_eff:])

    # ======================== Console report ========================
    print("\n" + "=" * 90)
    print(f"[Global Clustering] reference_word='{args.reference_word}' | n_clusters={args.n_clusters}")
    print("-" * 90)
    print("Cluster stats (sorted by mean_residual):")
    
    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster"])
        n_tok = int(row["n_tokens"])
        mean_res = float(row["mean_residual"])
        sub = df[df["cluster"] == cid].sort_values("count", ascending=False)
        top_tokens = [clean_token(t) for t in sub["token"].head(8).tolist()]
        top_str = ", ".join(top_tokens)
        print(f"  - cluster {cid:>2} | n_tokens={n_tok:>3} | mean_residual={mean_res:>8.4f} | top: {top_str}")
    
    print("-" * 90)
    if k_eff == 0:
        print("Removed clusters: none (k_each_side too small or not enough clusters)")
    else:
        low_removed = int(ordered[0])
        high_removed = int(ordered[-1])
        print(f"Removed clusters (extremes, k_each_side={args.k_each_side} -> k_eff={k_eff}): "
            f"lowest={low_removed}, highest={high_removed}")
    print("=" * 90 + "\n")

    df_kept = df[~df["cluster"].isin(remove_clusters)].copy()

    # ======================== 3) Intra-cluster CARE refine ========================
    tok2row = {tok: i for i, tok in enumerate(most_common_tokens)}
    df_kept["care_refine_remove"] = False

    for c_id in sorted(df_kept["cluster"].unique()):
        tokens_c = df_kept[df_kept["cluster"] == c_id]["token"].tolist()
        if len(tokens_c) < args.min_cluster_size:
            continue

        idxs = [tok2row[t] for t in tokens_c]
        W_c = W[idxs]

        sums = W_c.sum(dim=0, keepdim=True)
        P_minus = (sums - W_c) / (len(tokens_c) - 1)

        proj_orth = torch.eye(D) - PC
        D_minus = P_minus @ proj_orth.T
        d_norm2 = (D_minus ** 2).sum(dim=1)

        mean_excl = (d_norm2.sum() - d_norm2) / (len(tokens_c) - 1)
        thresh = (1.0 + args.alpha) * mean_excl

        remove_mask = d_norm2 > thresh
        for tok, rm in zip(tokens_c, remove_mask.tolist()):
            if rm:
                df_kept.loc[df_kept["token"] == tok,
                            "care_refine_remove"] = True

    df_final = df_kept[~df_kept["care_refine_remove"]].copy()

    # ======================== Console report ========================
    clean_tokens_final = [clean_token(t) for t in df_final["token"].tolist()]
    print("\n" + "=" * 90)
    print(f"[Final Kept Tokens] count={len(clean_tokens_final)}  | after intra-cluster refinement (CARE)")
    print("-" * 90)
    cols = 6
    for i, tok in enumerate(clean_tokens_final, 1):
        print(f"{tok:<18}", end="\n" if i % cols == 0 else "")
    if len(clean_tokens_final) % cols != 0:
        print()
    print("=" * 90 + "\n")

    # ======================== 4) JSON export ========================
    if args.prompt_style == "style":
        prompt_key = "painting"
        prefix = "A painting of "
    else:
        prompt_key = args.prompt_style
        prefix = "A photo of "

    clean_tokens = [clean_token(t) for t in df_final["token"].tolist()]

    base_words = [f"A {t}" for t in clean_tokens]
    prefixed_words = [f"{prefix}{t}" for t in clean_tokens]

    prefix_key = keyify(args.reference_word)

    out_json = {
        f"{prefix_key}_care_word": base_words,
        f"{prefix_key}_care_word_{prompt_key}": prefixed_words
    }

    json_path = os.path.join(
        args.output_dir,
        f"careset.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"[Saved] {json_path}")


# ------------------ Entry ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_word", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)

    parser.add_argument("--prompt_style", type=str, default="photo",
                        help="photo | style")

    parser.add_argument("--output_dir", type=str, default="data/nudity/")
    parser.add_argument("--vocab_cache", type=str,
                        default="./cache/clip_vocab_fp16.npy")

    parser.add_argument("--top_k_per_image", type=int, default=50)
    parser.add_argument("--top_n_tokens", type=int, default=100)

    parser.add_argument("--batch_size_txt", type=int, default=2048)
    parser.add_argument("--chunk_size_sim", type=int, default=4096)

    parser.add_argument("--n_clusters", type=int, default=7)
    parser.add_argument("--k_each_side", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
