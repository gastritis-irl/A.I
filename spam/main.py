import os
import re
import random
import math
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import KFold

from nb_clf import NaiveBayesClassifier

# Betöltjük az adathalmazt
data_dir = 'enron6'
ham_dir = os.path.join(data_dir, 'ham')
spam_dir = os.path.join(data_dir, 'spam')

# Betöltjük a stopszavakat
with open('stopwords.txt', 'r') as f:
    stopwords = set(line.strip() for line in f)

# Betöltjük a tanító és teszt adatok listáját
with open('train.txt', 'r') as f:
    train_files = [os.path.basename(line.strip()) for line in f]
with open('test.txt', 'r') as f:
    test_files = [os.path.basename(line.strip()) for line in f]


def preprocess(file_path, s_words):
    with open(file_path, 'r', errors='ignore') as f:
        content = f.read().lower()
    wordss = re.findall(r'\w+', content)
    return [word for word in wordss if word not in s_words]


# Tokenizálás, kisbetűsítés és stopszavak kiszűrése
ham_data = [preprocess(os.path.join(ham_dir, file), stopwords) for file in train_files if 'ham' in file]
spam_data = [preprocess(os.path.join(spam_dir, file), stopwords) for file in train_files if 'spam' in file]

# Adatok előkészítése
train_X = ham_data + spam_data
train_y = ['ham'] * len(ham_data) + ['spam'] * len(spam_data)
test_X = [preprocess(os.path.join(ham_dir if 'ham' in file else spam_dir, file), stopwords) for file in test_files]
test_y = ['ham' if 'ham' in file else 'spam' for file in test_files]

# A modell tanítása és tesztelése
for alpha in [0.01, 0.1, 1]:
    nb = NaiveBayesClassifier(alpha=alpha)
    nb.fit(train_X, train_y)
    accuracy = nb.test(test_X, test_y)
    print(f"Alpha: {alpha}, Accuracy: {accuracy}")

fp_rate, fn_rate = nb.calc_fp_fn(test_X, test_y)
print(f"False positive rate: {fp_rate}, False negative rate: {fn_rate}")


def cross_val_score(X, y, alpha, n_splits=5):
    kf = KFold(n_splits=n_splits)
    scores = []
    for train_index, test_index in kf.split(X):
        train_X, test_X = [X[i] for i in train_index], [X[i] for i in test_index]
        train_y, test_y = [y[i] for i in train_index], [y[i] for i in test_index]
        nb = NaiveBayesClassifier(alpha=alpha)
        nb.fit(train_X, train_y)
        scores.append(nb.test(test_X, test_y))
    return np.mean(scores)


alphas = [0.01, 0.1, 1]
cv_scores = [cross_val_score(train_X, train_y, alpha) for alpha in alphas]
best_alpha = alphas[np.argmax(cv_scores)]
print(f"Best alpha: {best_alpha}")

# ssl.zip adathalmaz betöltése
ssl_dir = 'ssl'
ssl_files = os.listdir(ssl_dir)
ssl_data = [preprocess(os.path.join(ssl_dir, file), stopwords) for file in ssl_files]

# Félig felügyelt tanulás
threshold = 5
converged = False
nb = NaiveBayesClassifier(alpha=best_alpha)
nb.fit(train_X, train_y)

while not converged:
    converged = True
    new_data = []
    new_labels = []
    for words in list(ssl_data):  # Create a copy for iteration
        log_prob_ham = nb.log_class_priors['ham'] + sum(nb.log_likelihoods['ham'][word] for word in words)
        log_prob_spam = nb.log_class_priors['spam'] + sum(nb.log_likelihoods['spam'][word] for word in words)
        if log_prob_ham - log_prob_spam >= math.exp(threshold):
            new_data.append(words)
            new_labels.append('ham')
            ssl_data.remove(words)
            converged = False
        elif log_prob_spam - log_prob_ham >= math.exp(threshold):
            new_data.append(words)
            new_labels.append('spam')
            ssl_data.remove(words)
            converged = False

    train_X += new_data
    train_y += new_labels
    nb.fit(train_X, train_y)

# Tesztelés újra
accuracy = nb.test(test_X, test_y)
print(f"Accuracy after semi-supervised learning: {accuracy}")
