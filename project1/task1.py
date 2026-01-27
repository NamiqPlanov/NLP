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
all_tokens = []
for t in texts:
    all_tokens.extend(tokenize(t))

tokenCount = len(all_tokens)                 
typeCount = len(set(all_tokens))              
freq = Counter(all_tokens) 
print("Number of tokens:", tokenCount)
print("Number of types:", typeCount)
print("Top 30 frequent words:", freq.most_common(30)) 


Ns = []
Vs = []
seen = set()
N = 0
for tok in all_tokens:
    N += 1
    seen.add(tok)
    Ns.append(N)
    Vs.append(len(seen))
logN = [math.log(n) for n in Ns if n > 0]
logV = [math.log(v) for v in Vs if v > 0]

n = len(logN)
sum_x = sum(logN)
sum_y = sum(logV)
sum_xx = sum(x*x for x in logN)
sum_xy = sum(x*y for x,y in zip(logN,logV))

beta = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x**2)
logk = (sum_y - beta*sum_x) / n
k = math.exp(logk)

print("Heaps Law k:", k)
print("Heaps Law beta:", beta)


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in v_in:
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = v_in[word]
    return v_out
vocab = Counter()
for w in all_tokens:
    vocab[" ".join(list(w)) + " </w>"] += 1

num_merges = 100
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

print("BPE Vocabulary size:", len(vocab))


