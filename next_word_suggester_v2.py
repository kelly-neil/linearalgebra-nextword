"""Minimal keyboard-style next-word suggester. Only needs NumPy."""
import glob
import math
import re
import sys
import numpy as np

# ---------------------------------------------------------------- corpus
corpus_files = sorted(glob.glob("corpus.txt") + glob.glob("corpus_*.txt"))

word_counts = {}
words = []


def read_corpus(corpus):
    for line in corpus:
        for w in re.findall(r"[a-z]+", line.lower()):
            word_counts[w] = word_counts.get(w, 0) + 1
            words.append(w)


if corpus_files:
    for path in corpus_files:
        with open(path, "r") as corpus_file:
            read_corpus(corpus_file)
        print(f"Corpus file read: {path}")
else:
    sys.exit("You need a corpus file ('corpus_[...].txt') first before using the program.")

if not word_counts:
    sys.exit("Corpus files contain no words.")

# --- vocabulary ---
vocab = sorted(word_counts)
idx = {w: i for i, w in enumerate(vocab)}
V, d, CTX = len(vocab), 16, 2          # vocab size, embedding dim, context length

freq = np.array([word_counts[w] for w in vocab], dtype=np.float32)
freq_probs = freq / freq.sum()

# --- training pairs: (last CTX words) -> next word ---
tokens = np.array([idx[w] for w in words], dtype=np.int32)
if len(tokens) > CTX:
    contexts = np.stack([tokens[i:len(tokens) - CTX + i] for i in range(CTX)], axis=1)
    targets = tokens[CTX:]
else:
    contexts = np.empty((0, CTX), dtype=np.int32)
    targets = np.empty((0,), dtype=np.int32)

rng = np.random.default_rng(0)
E = rng.normal(0, 0.1, (V, d)).astype(np.float32)   # embedding matrix
W = rng.normal(0, 0.1, (V, d)).astype(np.float32)   # output matrix


def softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


def predict_probs(context_ids):
    if not context_ids:
        return freq_probs.copy(), np.zeros(d, dtype=np.float32)
    h = E[context_ids].mean(axis=0)     # average the context embeddings
    return softmax(W @ h), h            # scores -> probabilities


# --- training: mini-batched gradient descent on cross-entropy ---
def train(E, W):
    if len(targets) == 0:
        return
    if V <= 512:
        batch_size = min(256, len(targets))
        lr = np.float32(0.2)
        for _ in range(200):
            perm = rng.permutation(len(targets))
            for start in range(0, len(targets), batch_size):
                b = perm[start:start + batch_size]
                ctx_b, tgt_b = contexts[b], targets[b]
                H = E[ctx_b].mean(axis=1)
                logits = H @ W.T
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                dlogits = probs
                dlogits[np.arange(len(b)), tgt_b] -= 1.0
                dH = dlogits @ W
                W -= lr * (dlogits.T @ H)
                for col in range(CTX):
                    np.add.at(E, ctx_b[:, col], -lr * dH / CTX)
    else:
        # Sampled softmax with AdaGrad for large vocabularies
        q = (freq ** 0.75).astype(np.float64)
        q /= q.sum()
        log_q = np.log(q).astype(np.float32)
        G_E = np.full((V, d), 1e-4, dtype=np.float32)
        G_W = np.full((V, d), 1e-4, dtype=np.float32)
        batch_size, num_neg, lr = 2048, 64, np.float32(0.2)
        n_batches = (len(targets) + batch_size - 1) // batch_size
        for _ in range(2):
            perm = rng.permutation(len(targets))
            neg_all = rng.choice(V, size=(n_batches, num_neg), replace=True, p=q)
            for bi, start in enumerate(range(0, len(targets), batch_size)):
                b = perm[start:start + batch_size]
                ctx_b, tgt_b = contexts[b], targets[b]
                neg = neg_all[bi]

                H = E[ctx_b].mean(axis=1)
                W_tgt, W_neg = W[tgt_b], W[neg]
                pos_logits = np.sum(H * W_tgt, axis=1, keepdims=True) - log_q[tgt_b][:, None]
                neg_logits = (H @ W_neg.T) - log_q[neg][None, :]
                neg_logits[neg[None, :] == tgt_b[:, None]] = -1e9
                logits = np.concatenate([pos_logits, neg_logits], axis=1)

                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                dlogits = probs
                dlogits[:, 0] -= 1.0

                dpos, dneg = dlogits[:, :1], dlogits[:, 1:]
                dH = dpos * W_tgt + dneg @ W_neg

                g_tgt = dpos * H
                np.add.at(G_W, tgt_b, g_tgt * g_tgt)
                np.add.at(W, tgt_b, -lr * g_tgt / np.sqrt(G_W[tgt_b]))

                g_neg = dneg.T @ H
                np.add.at(G_W, neg, g_neg * g_neg)
                np.add.at(W, neg, -lr * g_neg / np.sqrt(G_W[neg]))

                g_ctx = dH / CTX
                for col in range(CTX):
                    c = ctx_b[:, col]
                    np.add.at(G_E, c, g_ctx * g_ctx)
                    np.add.at(E, c, -lr * g_ctx / np.sqrt(G_E[c]))


train(E, W)


def char_bigram_vector(word):
    """Sparse vector of character-bigram counts, with ^ and $ boundary marks."""
    s = "^" + word + "$"
    v = {}
    for k in range(len(s) - 1):
        bg = s[k:k + 2]
        v[bg] = v.get(bg, 0) + 1
    return v


def cosine(u, v):
    dot = sum(u[k] * v[k] for k in u if k in v)
    nu = math.sqrt(sum(x * x for x in u.values()))
    nv = math.sqrt(sum(x * x for x in v.values()))
    return dot / (nu * nv) if nu and nv else 0.0


BIGRAM_VECS = {w: char_bigram_vector(w) for w in vocab}


def autocorrect(typed, k=1):
    tv = char_bigram_vector(typed)
    ranked = sorted(vocab, key=lambda w: cosine(tv, BIGRAM_VECS[w]), reverse=True)
    return ranked[:k]


def suggest(text, n=3):
    parts = text.lower().replace("\n", " ").split(" ")
    typed = parts[-1]                              # "" if text ends with a space
    context = [w for w in parts[:-1] if w]
    context_ids = [idx[w] for w in context[-CTX:] if w in idx]

    p, _ = predict_probs(context_ids)
    scores = p + 0.001 * freq_probs                # tiebreak with corpus frequency

    if typed:
        mask = np.fromiter((1.0 if w.startswith(typed) else 0.0 for w in vocab),
                           dtype=float, count=V)
        if mask.sum() > 0:
            scores = scores * mask                 # prefix filter: D p

    ranked = np.argsort(scores)[::-1]
    result = [vocab[j] for j in ranked if scores[j] > 0][:n]

    # autocorrect: if the typed word isn't a real word, put the best fix first
    if typed and typed not in idx:
        fix = autocorrect(typed)[0]
        if fix in result:
            result.remove(fix)
        result = ([fix] + result)[:n]
    return result


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    print(f"Vocabulary: {V} words. Type text; Enter on empty line to quit.")
    print("Tip: end with a space to see next-word suggestions.\n")
    while True:
        text = input("> ")
        if not text:
            break
        print("   suggestions:", suggest(text))