# Copyright (c) 2025 [MinaNadyFarag]. All rights reserved.
# This code is the property of [MinaNAdyFarag] and may not be copied, distributed, or used without permission.
# main.py

from data_preparation import tokenize, get_training_example
from model_architecture import forward_propagation
from model_training import train_model
from word_embeddings import extract_word_embeddings
from sentiment_analysis import perform_sentiment_analysis

# Example corpus
corpus = "I am happy because I am learning"

# Step 1: Data Preparation
words = tokenize(corpus)
word2Ind, Ind2word = get_dict(words)
V = len(word2Ind)

# Step 2: Model Training
W1, W2, b1, b2 = train_model(words, word2Ind, V)

# Step 3: Extract Word Embeddings
word_embeddings = extract_word_embeddings(W1, word2Ind)

# Step 4: Sentiment Analysis
perform_sentiment_analysis(word_embeddings, word2Ind)