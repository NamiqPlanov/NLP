import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from collections import defaultdict
import re


data = pd.read_csv("tales.csv")
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



