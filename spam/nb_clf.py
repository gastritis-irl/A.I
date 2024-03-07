import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.log_class_priors = None
        self.log_likelihoods = None

    def fit(self, X, y):
        num_docs = len(X)
        num_ham = sum(1 for label in y if label == 'ham')
        num_spam = num_docs - num_ham
        self.log_class_priors = {'ham': math.log(num_ham / num_docs), 'spam': math.log(num_spam / num_docs)}

        # Szószámlálás
        word_counts = {'ham': defaultdict(int), 'spam': defaultdict(int)}
        for words, label in zip(X, y):
            for word in words:
                word_counts[label][word] += 1

        # Valószínűségek számítása additív simítással
        self.log_likelihoods = {
            'ham': defaultdict(lambda: math.log(self.alpha / (num_ham + self.alpha * len(word_counts['ham'])))),
            'spam': defaultdict(lambda: math.log(self.alpha / (num_spam + self.alpha * len(word_counts['spam']))))}

        summ = 0
        for label, counts in word_counts.items():
            summ += sum(counts.values())
        for label, counts in word_counts.items():
            for word, count in counts.items():
                self.log_likelihoods[label][word] = math.log(
                    (count + self.alpha) / (sum(counts.values()) + self.alpha * summ))

    def predict(self, words):
        log_prob_ham = self.log_class_priors['ham'] + sum(self.log_likelihoods['ham'][word] for word in words)
        log_prob_spam = self.log_class_priors['spam'] + sum(self.log_likelihoods['spam'][word] for word in words)
        return 'ham' if log_prob_ham > log_prob_spam else 'spam'

    def test(self, X, y):
        correct = sum(1 for words, label in zip(X, y) if self.predict(words) == label)
        return correct / len(X)

    def calc_fp_fn(self, X, y):
        fp = fn = tp = tn = 0
        for words, label in zip(X, y):
            prediction = self.predict(words)
            if prediction == 'spam' and label == 'ham':
                fp += 1
            elif prediction == 'ham' and label == 'spam':
                fn += 1
            elif prediction == 'ham' and label == 'ham':
                tn += 1
            elif prediction == 'spam' and label == 'spam':
                tp += 1

        epsilon = 1e-10
        fp_rate = fp / (fp + tn + epsilon)
        fn_rate = fn / (fn + tp + epsilon)

        return fp_rate, fn_rate
