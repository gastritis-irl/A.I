import pandas as pd
from sklearn.metrics import accuracy_score
from classifiers.gradient_descent_linear_regression_binary_classifier import \
    GradientDescentLinearRegressionBinaryClassifier
from classifiers.exact_linear_regression_binary_classifier import ExactLinearRegressionBinaryClassifier
from classifiers.one_vs_one import OneVsOneLinearRegressionBinaryClassifier
from classifiers.one_vs_rest import OneVsRestLinearRegressionBinaryClassifier
import matplotlib.pyplot as plt

train_data = pd.read_csv('optdigits.tra', header=None)
test_data = pd.read_csv('optdigits.tes', header=None)

# Kiválasztott osztályok (pl. 4-esek és 7-esek)
selected_classes = [4, 7]

# Adatok előkészítése
train_data = train_data[train_data[64].isin(selected_classes)]
test_data = test_data[test_data[64].isin(selected_classes)]

x_train = train_data.iloc[:, :-1].values
y_train = (train_data.iloc[:, -1] == selected_classes[1]).astype(int)
x_test = test_data.iloc[:, :-1].values
y_test = (test_data.iloc[:, -1] == selected_classes[1]).astype(int)

# 1. Lineáris regresszió egzakt módszerrel bináris osztályozásra
exact_clf = ExactLinearRegressionBinaryClassifier()
exact_clf.fit(x_train, y_train)
y_pred_exact = exact_clf.predict(x_test)
accuracy_exact = accuracy_score(y_test, y_pred_exact)
print(f'Exact method accuracy: {accuracy_exact}')

# 2. Lineáris regresszió gradiens módszerrel bináris osztályozásra
gradient_clf = GradientDescentLinearRegressionBinaryClassifier()
gradient_clf.fit(x_train, y_train)
y_pred_gradient = gradient_clf.predict(x_test)
accuracy_gradient = accuracy_score(y_test, y_pred_gradient)
print(f'Gradient descent method accuracy: {accuracy_gradient}')

# 3.1. One vs Rest osztályozó
ovr_clf = OneVsRestLinearRegressionBinaryClassifier(GradientDescentLinearRegressionBinaryClassifier, list(range(10)))
ovr_clf.fit(x_train, y_train)
y_pred_ovr = ovr_clf.predict(x_test)
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
# print("One-vs-Rest classifier predictions:", y_pred_ovr)
print(f'1 vs Rest method accuracy: {accuracy_ovr}')

# 3.2. One vs One osztályozó
ovo_clf = OneVsOneLinearRegressionBinaryClassifier(GradientDescentLinearRegressionBinaryClassifier, list(range(10)))
ovo_clf.fit(x_train, y_train)
y_pred_ovo = ovo_clf.predict(x_test)
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
# print("One-vs-One classifier predictions:", y_pred_ovo)
print(f'1 vs 1 method accuracy: {accuracy_ovo}')

# 4. Tanulási görbe megjelenítése
loss_history = gradient_clf.get_loss_history()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')  # Mean Squared Error
# MSE Loss = (1/n) * Σ(Pi - Yi)²
plt.title('Learning Curve - Gradient Descent Method')
plt.show()
