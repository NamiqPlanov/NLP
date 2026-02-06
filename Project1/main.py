
import pandas as pd
import re
from collections import Counter, defaultdict
import math
import re
import csv
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("Project1/tales.csv")
texts = data["text"].dropna().astype(str).tolist()

AZ_LETTERS = r"A-Za-zƏəİıÖöÜüĞğŞşÇç"
TOKEN_REGEX = rf"[{AZ_LETTERS}]+(?:-\s*[{AZ_LETTERS}]+)*"

printer = "=" * 50

def save_tokens(tokens, path):
    with open(path, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_dictionary(freq, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tfrequency\n")
        for word, count in freq.most_common():
            f.write(f"{word}\t{count}\n")


# Task 1
def normalize(text):
    text = re.sub(
        rf"([{AZ_LETTERS}])-+\s+([{AZ_LETTERS}])",
        r"\1\2",
        text
    )
    return text

def tokenize(text):
    return re.findall(TOKEN_REGEX, normalize(text.lower()))

allTokens = []
for t in texts:
    allTokens.extend(tokenize(t))

tokenCount = len(allTokens)                 
typeCount = len(set(allTokens))              
freq = Counter(allTokens) 

save_tokens(allTokens, "Project1/space_based_tokens.txt")
save_dictionary(freq, "Project1/dictionary.tsv")

print(printer)
print("1. Tokenization")
print(printer)

print("Number of tokens:", tokenCount)
print("Number of types:", typeCount)
print("Top 50 frequent words:", freq.most_common(50))

# Task 2
Ns = []
Vs = []
seen = set()
N = 0
for tok in allTokens:
    N += 1
    seen.add(tok)
    Ns.append(N)
    Vs.append(len(seen))
logN = [math.log(n) for n in Ns if n > 0]
logV = [math.log(v) for v in Vs if v > 0]

n = len(logN)
sumX = sum(logN)
sumY = sum(logV)
sumXx = sum(x*x for x in logN)
sumXy = sum(x*y for x,y in zip(logN,logV))

beta = (n*sumXy - sumX*sumY) / (n*sumXx - sumX**2)
logk = (sumY - beta*sumX) / n
k = math.exp(logk)

print(printer)
print("2. Heaps' Law")
print(printer)

print("Heaps Law k:", k)
print("Heaps Law beta:", beta)

y_hat = [beta * x + logk for x in logN]

plt.figure()
plt.scatter(logN, logV)
plt.plot(logN, y_hat)

plt.xlabel("log(N)")
plt.ylabel("log(V)")
plt.title("Heaps' Law (log-log plot)")

plt.show()

# Task 3

print(printer)
print("3. BPE Tokenization")
print(printer)

def buildVocabulary(words):
    vocab = Counter()
    for word in words:
        chars = " ".join(list(word)) + " </w>"
        vocab[chars] += 1
    return vocab
vocab = buildVocabulary(allTokens)

def getPairStats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def mergePair(pair, vocab):
    newVocabulary = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in vocab:
        newWord = re.sub(rf'(?<!\S){bigram}(?!\S)', replacement, word)
        newVocabulary[newWord] = vocab[word]
    return newVocabulary

def trainBpe(vocab, numMerges=1000):
    merges = []
    for i in range(numMerges):
        pairs = getPairStats(vocab)
        if not pairs:
            break
        bestPair = max(pairs, key=pairs.get)
        vocab = mergePair(bestPair, vocab)
        merges.append(bestPair)
        if i % 100 == 0:
            print(f"Merge {i}: {bestPair}")
    return vocab, merges
numMerges = 1000  
finalVocab, merges = trainBpe(vocab, numMerges)

def bpeTokenizeWord(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(tokens)-1:
            if tokens[i] == a and tokens[i+1] == b:
                tokens[i:i+2] = [a+b]
            else:
                i += 1
    return tokens

def bpeTokenizeText(text, merges):
    words = tokenize(text)
    bpeTokens = []
    for w in words:
        bpeTokens.extend(bpeTokenizeWord(w, merges))
    return bpeTokens

allBpeTokens = []
for t in texts:
    allBpeTokens.extend(bpeTokenizeText(t, merges))

bpeTokenFreq = Counter(allBpeTokens)

# save_tokens(allBpeTokens, "bpe_tokens.txt")
# save_dictionary(bpeTokenFreq, "bpe_dictionary.tsv")

print("Total BPE tokens:", len(allBpeTokens))
print("BPE vocabulary size:", len(set(allBpeTokens)))
print("\nTop 50 BPE tokens:\n{}".format(bpeTokenFreq.most_common(50)))

# Task 4
def sentenceSegmentAZ(text):
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text)

    text = text.replace("“", '"').replace("”", '"')

    text = re.sub(r'(^|\n)\s*[-–—]\s*', r'\1— ', text)
    text = re.sub(r'\s+[-–—]\s+', ' — ', text)

    text = re.sub(r'([.!?])(?=[A-ZƏÖÜĞİŞÇ])', r'\1 ', text)

    az_abbr = ["məs","və s","b.e","prof","dr","akad","dos","müəl","müh","müd","ş"]

    for abbr in az_abbr:
        text = re.sub(rf'\b{abbr}\.', f'{abbr}<DOT>', text, flags=re.IGNORECASE)

    text = re.sub(r':\s*[-–—]?\s*', ':\n— ', text)

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-ZƏÖÜİĞŞÇ—])|'
        r'(?<=[.!?])\s*(?=—)|'
        r'\n+',
        text
    )

    clean = []
    for s in sentences:
        s = s.replace("<DOT>", ".").strip()
        if re.search(r'[A-Za-zƏÖİÜĞŞÇıəöüğşç0-9]', s):
            clean.append(s)

    return clean

print(printer)
print("4. Sentence Segmentation")
print(printer)

all_sentences = []

for text in texts:
    sentences = sentenceSegmentAZ(text)
    all_sentences.extend(sentences)

save_tokens(all_sentences, 'Project1/sentence_segmentation.txt')

for s in all_sentences[:20]:
    print(s)



# Task 5
# Task 5: Spell Checking (Fixed)
# ===============================

def levenshtein(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 2
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost  # substitute / match
            )
    return dp


def backtrace(a, b):
    dp = levenshtein(a, b)
    i, j = len(a), len(b)
    operations = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 2
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    operations.append(("match", a[i - 1], b[j - 1]))
                else:
                    operations.append(("substitute", a[i - 1], b[j - 1]))
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(("delete", a[i - 1], "-"))
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append(("insert", "-", b[j - 1]))
            j -= 1
            continue

        raise RuntimeError("Backtrace failed")

    operations.reverse()
    return operations, dp[-1][-1]


# Dictionary (single-word tokens)
dictionary = set(allTokens)


def spellCheckWord(word, k=3, max_dist=2):
    candidates = []

    for w in dictionary:
        ops, d = backtrace(word, w)
        if d <= max_dist:
            candidates.append({
                "word": w,
                "distance": d,
                "ops": ops
            })

    candidates.sort(key=lambda x: x["distance"])
    return candidates[:k]


def spellCheckPhrase(phrase, k=3, max_dist=4):
    words = phrase.split()
    corrected = []
    all_suggestions = []

    for w in words:
        suggestions = spellCheckWord(w.lower(), k=k, max_dist=max_dist)
        if suggestions:
            best = suggestions[0]["word"]
            corrected.append(best)
            all_suggestions.append((w, suggestions))
        else:
            corrected.append(w)
            all_suggestions.append((w, []))

    return " ".join(corrected), all_suggestions


def format_ops(ops):
    out = []
    for op in ops:
        if op[0] == "match":
            out.append(op[1])
        elif op[0] == "substitute":
            out.append(f"{op[1]}→{op[2]}")
        elif op[0] == "insert":
            out.append(f"+{op[2]}")
        elif op[0] == "delete":
            out.append(f"-{op[1]}")
    return " ".join(out)


def print_spellcheck_phrase(input_phrase, gold):
    corrected, suggestions = spellCheckPhrase(input_phrase)

    print(f"Input phrase   : {input_phrase}")
    print(f"Gold standard  : {gold}")
    print(f"Corrected text : {corrected}")
    print("")

    for word, suggs in suggestions:
        print(f"Word: '{word}'")
        if not suggs:
            print("  No suggestions")
        else:
            for i, s in enumerate(suggs, 1):
                print(f"  {i}. {s['word']} (d={s['distance']})")
                print(f"     edits: {format_ops(s['ops'])}")
        print()

print("=" * 50)
print("5. Spell Checking with Levenshtein Distance")
print("=" * 50)
while True:
    phrase = input("Enter text (or press Enter to exit): ").strip()
    if not phrase:
        break

    corrected, suggestions = spellCheckPhrase(phrase)

    print("\nCorrected text:", corrected)
    print("")

    for word, suggs in suggestions:
        print(f"Word: '{word}'")
        if not suggs:
            print("  No suggestions")
        else:
            for i, s in enumerate(suggs, 1):
                print(f"  {i}. {s['word']} (d={s['distance']})")
                print(f"     edits: {format_ops(s['ops'])}")
        print()

# Additional Task
AZ_ALPHABET = list("abcçdeəfgğhxıijkqlmnoöprsştuüvyz")
AZ_SET = set(AZ_ALPHABET)

def is_az_word(word):
    return word and all(c in AZ_SET for c in word.lower())

def char_ngrams(word, n=2):
    return {word[i:i+n] for i in range(len(word) - n + 1)}

def load_dictionary(path):
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if is_az_word(w):
                words.add(w)
    print(f"Dictionary loaded: {len(words)} words")
    return words

def load_tokens(path):
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip().lower()
            if is_az_word(t):
                tokens.append(t)
    print(f"Tokens loaded: {len(tokens)}")
    return tokens


# Computational Complexity: O(D⋅L); D = |dictionary|, L = average word length
def build_ngram_index(dictionary, n=2):
    index = defaultdict(set)
    for w in dictionary:
        for ng in char_ngrams(w, n):
            index[ng].add(w)
    return index

def levenshtein_distance(a, b, max_dist):
    n, m = len(a), len(b)
    if abs(n - m) > max_dist:
        return max_dist + 1

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [max_dist + 1] * m
        row_min = curr[0]

        j_start = max(1, i - max_dist)
        j_end   = min(m, i + max_dist)

        for j in range(j_start, j_end + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j-1] + 1,
                prev[j-1] + cost
            )
            row_min = min(row_min, curr[j])

        if row_min > max_dist:
            return max_dist + 1
        prev = curr

    return prev[m]

def levenshtein_subs(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    ptr = [[None]*(m+1) for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = i
        ptr[i][0] = 'up'
    for j in range(m+1):
        dp[0][j] = j
        ptr[0][j] = 'left'
    ptr[0][0] = None

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            choices = [
                (dp[i-1][j] + 1, 'up'),
                (dp[i][j-1] + 1, 'left'),
                (dp[i-1][j-1] + cost, 'diag')
            ]
            dp[i][j], ptr[i][j] = min(choices, key=lambda x: x[0])

    i, j = n, m
    subs = []
    while i > 0 and j > 0:
        if ptr[i][j] == 'diag':
            if a[i-1] != b[j-1]:
                subs.append((a[i-1], b[j-1]))
            i -= 1
            j -= 1
        elif ptr[i][j] == 'up':
            i -= 1
        else:
            j -= 1

    return subs, dp[n][m]

# Computational Complexity: O(t⋅C); t = length of token, C = average number of dictionary words sharing a bigram
def get_candidates(token, ngram_index, min_shared=2):
    grams = char_ngrams(token)
    scores = Counter()

    for g in grams:
        for w in ngram_index.get(g, []):
            scores[w] += 1

    return [w for w,c in scores.items() if c >= min_shared]

def build_confusion_matrix(tokens, dictionary, max_dist=2):
    ngram_index = build_ngram_index(dictionary)
    confusions = Counter()
    skipped = 0

    for token in tokens:
        candidates = get_candidates(token, ngram_index)
        if not candidates:
            skipped += 1
            continue

        best = None
        best_d = max_dist + 1

        for w in candidates:
            d = levenshtein_distance(w, token, max_dist)
            if d < best_d:
                best = w
                best_d = d

        if not best or best_d > max_dist:
            skipped += 1
            continue

        subs, _ = levenshtein_subs(best, token)
        for a, b in subs:
            if a in AZ_SET and b in AZ_SET:
                confusions[(a, b)] += 1
        print(token, best, a, b, confusions[(a, b)])
    print(f"Skipped tokens: {skipped}")
    return confusions

def export_confusion_matrix(confusions, path):
    matrix = {
        c: {t: 0 for t in AZ_ALPHABET}
        for c in AZ_ALPHABET
    }

    for (correct, typed), cnt in confusions.items():
        matrix[correct][typed] += cnt

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["correct \\ typed"] + AZ_ALPHABET)
        for c in AZ_ALPHABET:
            writer.writerow([c] + [matrix[c][t] for t in AZ_ALPHABET])

    print(f"Confusion matrix saved to {path}")

# dictionary = load_dictionary("/content/sample_data/az.txt")
# tokens = load_tokens("/content/sample_data/noisy_tokens.txt")

# confusions = build_confusion_matrix(tokens, dictionary)
# export_confusion_matrix(confusions, "az_confusion_matrix.csv")

# print("\nTop substitutions:")
# for (a,b),c in confusions.most_common(20):
#     print(f"{a} → {b}: {c}")

conf_matrix = {c:{w:0 for w in AZ_ALPHABET} for c in AZ_ALPHABET}

with open("Project1/az_confusion_matrix.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        correct = row[0]
        for typed, val in zip(AZ_ALPHABET, row[1:]):
            conf_matrix[correct][typed] = int(val)

weights = {}
for correct in AZ_ALPHABET:
    for typed in AZ_ALPHABET:
        if correct == typed:
            weights[(typed, correct)] = 0
        else:
            count = conf_matrix[correct][typed]
            weights[(typed, correct)] = 1 / (count + 1)

def weightedLevenshtein(a, b, weights=None):
    n = len(a)
    m = len(b)
    dp = np.zeros((n+1, m+1))

    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost_sub = 0 if a[i-1] == b[j-1] else 1
            if weights and (a[i-1], b[j-1]) in weights:
                cost_sub = weights[(a[i-1], b[j-1])]
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost_sub
            )
    return dp[n][m]

dictionary = set(allTokens)

def spellCheck(word, weights=None, dictionary=dictionary):
    best_score = float('inf')
    best_word = word
    for v in dictionary:  
        score = weightedLevenshtein(word, v, weights)  
        if score < best_score:
            best_score = score
            best_word = v
    return best_word

typoExamples = {
    "salam": "sələm",
    "gördüm": "gordum",
    "kitab": "kitap",
    "qız": "qiz",
    "ev": "ew"
}
print(printer)
print("Additional Task. Weighted Levenshtein Spell Checker")
print(printer)
for correct, typo in typoExamples.items():
    best_score = float('inf')
    best_word = typo
    for word in dictionary:
        dist = weightedLevenshtein(typo, word, weights)
        if dist < best_score:
            best_score = dist
            best_word = word
    print(f"Input: {typo} -> Corrected: {best_word} | Weighted distance: {best_score} | Gold: {correct}")
