import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


# Preprocessing function
def preprocess_text(text):
    """
    Function to preprocess text: tokenization, convert to lower case,
    remove punctuation, numbers, and stop words.
    """
    word_tokens = word_tokenize(text)
    word_tokens = [word.lower() for word in word_tokens if word.isalpha() and word not in stop_words]
    return ' '.join(word_tokens)


# 2. Load dataset
data = pd.read_csv('Youtube01-Psy.csv')
X = data['CONTENT'].fillna(' ').apply(str).apply(preprocess_text)  # preprocessing
y = data['CLASS']

# 1. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    # List of classifiers to be trained
    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('RadiusNeighborsClassifier', RadiusNeighborsClassifier(outlier_label=1)),
    ('MultinomialNB', MultinomialNB()),
    ('ComplementNB', ComplementNB()),
    ('SGDClassifier log', SGDClassifier(loss='log_loss')),
    ('SGDClassifier hinge', SGDClassifier(loss='hinge')),
]

# 6. Loop for n-grams
for ngram_range in [(1, 1), (1, 2), (1, 3)]:  # unigrams, bigrams, trigrams
    # 3. Vectorization / Feature extraction
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    for name, model in models:
        # 4. Model training
        model.fit(X_train_transformed, y_train)
        # 5. Predictions and performance measurement
        predicted = model.predict(X_test_transformed)
        ngram_label = 'unigrams' if ngram_range == (1, 1) else ('bigrams' if ngram_range == (1, 2) else 'trigrams')
        print(f'{name} with {ngram_label} accuracy: {accuracy_score(y_test, predicted)}')
