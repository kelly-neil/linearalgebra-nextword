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

# ---------------------------------------------------------------- corpus
corpus_file = open("corpus.txt", "r")

if corpus_file:
    CORPUS = corpus_file.read()
    print("Corpus file read.")
else:
    print("Using default corpus text.")
    CORPUS = """
    i like to drink coffee in the morning. i like to drink tea in the evening.
    i like to read a book in the evening. she likes to drink coffee with milk.
    the cat sat on the mat. the dog sat on the floor. the cat likes to drink milk.
    i went to the store to buy some milk. i went to the park to read a book.
    she went to the store to buy some coffee. we like to go to the park.
    they like to read the news in the morning. the morning is cold today.
    i receive the news every morning. she will receive the book today.
    i think the weather is nice today. the weather is cold in the morning.
    we will go to the store today. they will go to the park in the evening.
    """



# Get entire vocabulary found in the corpus
words = re.findall(r"[a-z]+", CORPUS.lower())
vocab = sorted(set(words))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# ------------------------------------------- bigram matrix B -> P (sparse)
B = {}                                   # B[i][j] = count of (word i -> word j)
for a, b in zip(words, words[1:]):
    B.setdefault(idx[a], {}).setdefault(idx[b], 0)
    B[idx[a]][idx[b]] += 1

P = {}                                   # row-normalized: each row sums to 1
for i, row in B.items():
    total = sum(row.values())
    P[i] = {j: c / total for j, c in row.items()}

freq = [0] * V                           # unigram counts, used as a tiny tiebreaker
for w in words:
    freq[idx[w]] += 1


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


def autocorrect(typed, k=1):
    tv = char_bigram_vector(typed)
    ranked = sorted(vocab, key=lambda w: cosine(tv, BIGRAM_VECS[w]), reverse=True)
    return ranked[:k]


# --------------------------------------------------------- suggestion logic
def suggest(text, n=3):
    parts = text.lower().split(" ")
    typed = parts[-1]                              # "" if text ends with a space
    context = [w for w in parts[:-1] if w]

    # scores = D (c^T P)
    scores = vec_times_matrix(context_vector(context), P)
    total_freq = sum(freq)
    scores = [s + 0.001 * f / total_freq for s, f in zip(scores, freq)]  # tiebreak

    mask = [1 if w.startswith(typed) else 0 for w in vocab]              # diag of D
    if typed and sum(mask) > 0:
        scores = diag_times_vector(mask, scores)

    ranked = sorted(range(V), key=lambda j: scores[j], reverse=True)
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
