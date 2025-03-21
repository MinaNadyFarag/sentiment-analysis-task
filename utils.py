# Copyright (c) 2025 [MinaNadyFarag]. All rights reserved.
# This code is the property of [MinaNAdyFarag] and may not be copied, distributed, or used without permission.
# utils.py

def get_dict(words):
    word2Ind = {}
    Ind2word = {}
    for idx, word in enumerate(sorted(set(words))):
        word2Ind[word] = idx
        Ind2word[idx] = word
    return word2Ind, Ind2word