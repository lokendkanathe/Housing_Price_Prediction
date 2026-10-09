import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../data/housing.csv")
print("Dataset Shap:", df.shape)

df["date"] = pd.to_datetime(df["date"])
df["sale_year"] = df["date"].dt.year
df["house_age"] = df["sale_year"] - df["yr_built"]

city_encoder = LabelEncoder()
df["city_encoded"] = city_encoder.fit_transform(df["city"])

joblib.dump(city_encoder, "../models/city_encoder.pkl")

X = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "house_age",
        "city_encoded"
    ]
]

y=df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lr=LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
rf_r2 = r2_score(y_test, rf_pred)

if rf_r2 > lr_r2:
    best_model = rf
    best_model_name = "Random Forest"
else:
    best_model = lr
    best_model_name = "Linear Regression"

joblib.dump(best_model, "../models/house_price_model.pkl")
print("\nModel saved successfully!")