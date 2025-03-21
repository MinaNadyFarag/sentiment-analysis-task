# Copyright (c) 2025 [MinaNadyFarag]. All rights reserved.
# This code is the property of [MinaNAdyFarag] and may not be copied, distributed, or used without permission.
# word_embeddings.py

import numpy as np

def extract_word_embeddings(W1, word2Ind):
    word_embeddings = W1.T
    for word, index in word2Ind.items():
        print(f"Word: {word}, Embedding: {word_embeddings[index]}")
    return word_embeddings