import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Adathalmaz betöltése
data = pd.read_csv("Real_estate.csv")

# Jellemzők (X1-X6) és célváltozó (Y house price of unit area) kiválasztása
X = data.iloc[:, 1:7]
y = data.iloc[:, 7]

# 3D ábra létrehozása
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Adatpontok ábrázolása
ax.scatter(X["X5 latitude"], X["X6 longitude"], y, c='blue', marker='o', alpha=0.5)

# Tengelyfeliratok hozzáadása
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('House Price')

plt.show()

# Adatok felosztása tanító és tesztelő halmazra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Jellemzők normalizálása
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection használata
k_best = SelectKBest(score_func=f_regression, k=3)
X_train_best = k_best.fit_transform(X_train_scaled, y_train)
X_test_best = k_best.transform(X_test_scaled)

# SGDRegressor modell létrehozása és illesztése
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, random_state=42)
sgd_regressor.fit(X_train_best, y_train)

# Tanulási és teszt hibák kiszámítása
y_train_pred_sgd = sgd_regressor.predict(X_train_best)
y_test_pred_sgd = sgd_regressor.predict(X_test_best)

train_error_sgd = mean_squared_error(y_train, y_train_pred_sgd)
test_error_sgd = mean_squared_error(y_test, y_test_pred_sgd)

# X5 és X6 oszlopok alapján történő regresszió és hibák kiszámítása
X_lat_long = X.iloc[:, 4:6]

X_train_lat_long, X_test_lat_long, y_train_lat_long, y_test_lat_long = train_test_split(X_lat_long, y, test_size=0.2,
                                                                                        random_state=42)

lr_lat_long = LinearRegression()
lr_lat_long.fit(X_train_lat_long, y_train_lat_long)

y_train_pred_lat_long = lr_lat_long.predict(X_train_lat_long)
y_test_pred_lat_long = lr_lat_long.predict(X_test_lat_long)

train_error_lat_long = mean_squared_error(y_train_lat_long, y_train_pred_lat_long)
test_error_lat_long = mean_squared_error(y_test_lat_long, y_test_pred_lat_long)

# Keresztvalidációs pontszámok kiszámítása
cv_scores_sgd = cross_val_score(sgd_regressor, X_train_best, y_train, cv=5, scoring="neg_mean_squared_error")
cv_scores_lat_long = cross_val_score(lr_lat_long, X_lat_long, y, cv=5, scoring="neg_mean_squared_error")

# Kiíratjuk az eredményeket
print("SGD Train error:", train_error_sgd)
print("SGD Test error:", test_error_sgd)
print("SGD Cross validation scores:", cv_scores_sgd)

print("Lat-Long Train error:", train_error_lat_long)
print("Lat-Long Test error:", test_error_lat_long)
print("Lat-Long Cross validation scores:", cv_scores_lat_long)

# Kiválasztott jellemzők kiíratása
selected_features = X.columns[k_best.get_support()]
print("Selected features:", selected_features)
