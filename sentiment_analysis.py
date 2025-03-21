# Copyright (c) 2025 [MinaNadyFarag]. All rights reserved.
# This code is the property of [MinaNAdyFarag] and may not be copied, distributed, or used without permission.
# sentiment_analysis.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def perform_sentiment_analysis(word_embeddings, word2Ind):
    # Example labeled data (word: sentiment)
    data = {
        "happy": 1,
        "sad": 0,
        "joy": 1,
        "angry": 0,
        "excited": 1,
        "bored": 0
    }

    # Prepare dataset
    X = np.array([word_embeddings[word2Ind[word]] for word in data.keys()])
    y = np.array([data[word] for word in data.keys()])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")