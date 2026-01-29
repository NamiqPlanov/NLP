import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from collections import defaultdict
import re
from sklearn.metrics import confusion_matrix


data = pd.read_csv("project1/tales.csv")
texts = data["text"].dropna().astype(str).tolist()

#task1
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zA-Zəğıöüşç]+", text)
    return tokens
allTokens = []
for t in texts:
    allTokens.extend(tokenize(t))

tokenCount = len(allTokens)                 
typeCount = len(set(allTokens))              
freq = Counter(allTokens) 
print("Number of tokens:", tokenCount)
print("Number of types:", typeCount)
print("Top 50 frequent words:", freq.most_common(50)) 

#task2
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
print("Heaps Law k:", k)
print("Heaps Law beta:", beta)




#task5
def levenshtein(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1,
                           dp[i][j-1]+1,
                           dp[i-1][j-1]+cost)
    return dp[-1][-1]
dictionary = set(allTokens)
def spellCheck(word, k=3):
    candidates = []
    for w in dictionary:
        d = levenshtein(word, w)
        if d <= 2:
            candidates.append((w,d))
    return sorted(candidates, key=lambda x: x[1])[:k]
print(spellCheck("qoymrd")) 


#task3
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
print("Total BPE tokens:", len(allBpeTokens))
print("BPE vocabulary size:", len(set(allBpeTokens)))
print("\nTop 50 BPE tokens:\n{}".format(bpeTokenFreq.most_common(50)))


#task4

def sentenceSegmentAZ(text):
    text = re.sub(r"\s+", " ", text)
    az_abbr = [
        "məs", "və s", "b.e", "prof", "dr", "akad", "dos",
        "müəl", "müh", "müd", "ş"
    ]
    for abbr in az_abbr:
        text = re.sub(rf"{abbr}\.", f"{abbr}<DOT>", text, flags=re.IGNORECASE)
    text = re.sub(r'([.!?])(?=–)', r'\1 ', text)
    text = text.replace("—", "<DASH>")  
    sentences = re.split(
        r'(?<=[.!?])\s+|(?<=:)(?=\s*–)|(?<=\.)\s+(?=Tülkü dedi:)',
        text
    )
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    sentences = [s.replace("<DASH>", "—") for s in sentences]
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences


test_text = "Padşah çağırdı oğlanı. Oğlan gəldi. Padşah dedi: — Harada idin? Oğlan cavab verdi: — Ovdaydım!"
sentences = sentenceSegmentAZ(test_text)

sentences = sentenceSegmentAZ(texts[0])
for s in sentences:
    print(s)


#Additional task

typoExamples = {
    "salam": "sələm",
    "gördüm": "gordum",
    "kitab": "kitap",
    "qız": "qiz",
    "ev": "ew"
}
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

confusionCounts = {}
for correct, typo in typoExamples.items():
    for w, c in zip(typo, correct):
        if w != c:
            confusionCounts[(w, c)] = confusionCounts.get((w, c), 0) + 1

weights = {k: 1/(v+1) for k,v in confusionCounts.items()}
print("Weighted substitution dictionary:", weights)
dictionary = set(allTokens)
def spellCheck(word, weights=None):
    best_score = float('inf')
    best_word = word
    for v in dictionary:  
        score = weightedLevenshtein(word, v, weights)  
        if score < best_score:
            best_score = score
            best_word = v
    return best_word

predicted = []
gold = []
for correct, typo in typoExamples.items():
    corrected = spellCheck(typo, weights)
    predicted.append(corrected)
    gold.append(correct)

allWords = list(set(predicted + gold))
word2idx = {w:i for i,w in enumerate(allWords)}
pred_idx = [word2idx[w] for w in predicted]
gold_idx = [word2idx[w] for w in gold]

cm = confusion_matrix(gold_idx, pred_idx)
print("\n===== SPELL CHECK RESULTS =====")
for t, p, g in zip(typoExamples.values(), predicted, gold):
    print(f"Input: {t} -> Corrected: {p} (Gold: {g})")

print("\n===== CONFUSION MATRIX =====")
print("Words:", allWords)
print(cm)