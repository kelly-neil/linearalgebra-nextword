"""
Next-Word Suggestion with Autocorrect  (plain Python, no libraries except re/math)

Linear algebra used:
  - Bigram count matrix B -> row-normalized transition matrix P
  - Vector-matrix product  c^T P   (c = decayed context vector)
  - Diagonal matrix D = diag(m) for prefix filtering
  - Cosine similarity of character-bigram vectors for autocorrect
"""
import re
import math
import glob
import sys

# ---------------------------------------------------------------- corpus
corpus_files = sorted(glob.glob("corpus.txt") + glob.glob("corpus_*.txt"))

word_counts = {}
word_bigrams = {}

def read_corpus(corpus):
    prev = None
    for line in corpus:
        for w in re.findall(r"[a-z]+", line.lower()):
            word_counts[w] = word_counts.get(w, 0) + 1
            if prev is not None:
                row = word_bigrams.setdefault(prev, {})
                row[w] = row.get(w, 0) + 1
            prev = w

if corpus_files:
    for path in corpus_files:
        with open(path, "r") as corpus_file:
            read_corpus(corpus_file)
        print(f"Corpus file read: {path}")
else:
    sys.exit("You need a corpus file ('corpus_[...].txt') first before using the program.")

# Get entire vocabulary found in the corpus
vocab = sorted(word_counts)
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# ------------------------------------------- bigram matrix B -> P (sparse)
B = {                                    # B[i][j] = count of (word i -> word j)
    idx[a]: {idx[b]: c for b, c in row.items()}
    for a, row in word_bigrams.items()
}

P = {}                                   # Normalize rows so that each row sums up to 1 (100%)
for i, row in B.items():
    total = sum(row.values())
    P[i] = {j: c / total for j, c in row.items()}

freq = [word_counts[w] for w in vocab]   # In the event that there is a tie in probability, use the frequency as a tie-breaker.


# ------------------------------------------------- linear algebra helpers
def context_vector(context_words, alpha=0.2):
    """c = e(w_t) + alpha*e(w_t-1) + alpha^2*e(w_t-2) + ...  (sparse dict)."""
    c, weight = {}, 1.0
    for w in reversed(context_words[-3:]):
        if w in idx:
            c[idx[w]] = c.get(idx[w], 0) + weight
        weight *= alpha
    return c


def vec_times_matrix(c, P):
    """Compute c^T P, where c is a sparse vector and P a sparse matrix."""
    out = [0.0] * V
    for i, ci in c.items():
        for j, pij in P.get(i, {}).items():
            out[j] += ci * pij
    return out


def diag_times_vector(m, x):
    """D x, where D = diag(m). Same as elementwise multiply."""
    return [mi * xi for mi, xi in zip(m, x)]


def char_bigram_vector(word):
    """Sparse vector of character-bigram counts, with ^ and $ boundary marks."""
    s = "^" + word + "$"
    v = {}
    for k in range(len(s) - 1):
        v[s[k:k + 2]] = v.get(s[k:k + 2], 0) + 1
    return v


def cosine(u, v):
    dot = sum(u[k] * v[k] for k in u if k in v)
    nu = math.sqrt(sum(x * x for x in u.values()))
    nv = math.sqrt(sum(x * x for x in v.values()))
    return dot / (nu * nv) if nu and nv else 0.0


BIGRAM_VECS = {w: char_bigram_vector(w) for w in vocab}

# ------------------------------------------------- autocorrect (returns multiple options)
def autocorrect(typed, k=3):
    tv = char_bigram_vector(typed)
    ranked = sorted(vocab, key=lambda w: cosine(tv, BIGRAM_VECS[w]), reverse=True)
    return ranked[:k]

# --------------------------------------------------------- suggestion logic
def suggest(text, n=3):
    parts = text.split(" ")  # Keep original casing check for sentence starts
    typed = parts[-1].lower()
    context = [w.lower() for w in parts[:-1] if w]
    
    # Check if we are at the start of a sentence (after period/empty)
    is_start_of_sentence = len(context) == 0 or (len(parts) > 1 and parts[-2].endswith('.'))

    # scores = D (c^T P)
    scores = vec_times_matrix(context_vector(context), P)
    total_freq = sum(freq)
    
    # Fallback/Tiebreak: Blend with global frequency so cold-starts still suggest common words
    scores = [s + 0.05 * (f / total_freq) for s, f in zip(scores, freq)]  

    mask = [1 if w.startswith(typed) else 0 for w in vocab]
    if typed and sum(mask) > 0:
        scores = diag_times_vector(mask, scores)

    ranked = sorted(range(V), key=lambda j: scores[j], reverse=True)
    result = [vocab[j] for j in ranked if scores[j] > 0][:n]

    # Autocorrect integration for out-of-vocabulary typos
    if typed and typed not in idx:
        fixes = autocorrect(typed, k=n)
        for fix in fixes:
            if fix not in result:
                result.insert(0, fix)
        result = result[:n]

    # Apply smart capitalization if starting a sentence
    if is_start_of_sentence:
        result = [w.capitalize() for w in result]

    return result
