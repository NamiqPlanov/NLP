import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from collections import defaultdict
import re


data = pd.read_csv("project1/tales.csv")
texts = data["text"].dropna().astype(str).tolist()

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
print("Top 30 frequent words:", freq.most_common(30)) 


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


def getStats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def mergeVocabulary(pair, vIn):
    vOut = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vIn:
        newWord = word.replace(bigram, replacement)
        vOut[newWord] = vIn[word]
    return vOut
vocab = Counter()
for w in allTokens:
    vocab[" ".join(list(w)) + " </w>"] += 1

num_merges = 100
for i in range(num_merges):
    pairs = getStats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = mergeVocabulary(best, vocab)

print("BPE Vocabulary size:", len(vocab))



def sentenceSegment(text):
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences
testText = texts[0]
sentences = sentenceSegment(testText)

for s in sentences[:10]:
    print(s)


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
confMatrix = defaultdict(int)
def weighted_levenshtein(a, b, weights):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            if a[i-1]==b[j-1]:
                cost = 0
            else:
                cost = weights.get((a[i-1], b[j-1]), 1)
                confMatrix[(a[i-1], b[j-1])] += 1

            dp[i][j] = min(dp[i-1][j]+1,
                           dp[i][j-1]+1,
                           dp[i-1][j-1]+cost)
    return dp[-1][-1]
