# ML_Prediction
Bike Rental Demand 
# 🚲 Bike Rental Demand Prediction

An end-to-end machine learning regression project for predicting hourly bike rental demand using weather, temporal, and seasonal information.

The project explores the Bike Sharing dataset, performs exploratory data analysis and feature engineering, and compares three regression algorithms:

- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression

The best-performing model in the experiments was **Random Forest Regression**, achieving an **R² score of approximately 0.902** on the test set.

---

## 📌 Project Overview

Bike-sharing systems generate large amounts of data about when and under what conditions bicycles are rented.

The objective of this project is to answer the following question:

> **Can machine learning predict hourly bike rental demand using information such as weather, temperature, season, hour of the day, and working-day status?**

This is formulated as a **supervised regression problem**, where the target variable is the total number of bike rentals during an hour.

---

## 📊 Dataset

The project uses the Bike Sharing dataset obtained from Hugging Face.

The dataset contains hourly bike-sharing observations with the following columns:

| Feature | Description |
|---|---|
| `instant` | Record identifier |
| `dteday` | Date |
| `season` | Season |
| `yr` | Year |
| `mnth` | Month |
| `hr` | Hour of the day |
| `holiday` | Whether the day is a holiday |
| `weekday` | Day of the week |
| `workingday` | Whether the day is a working day |
| `weathersit` | Weather condition |
| `temp` | Normalized temperature |
| `atemp` | Normalized feels-like temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `casual` | Number of casual users |
| `registered` | Number of registered users |
| `cnt` | Total number of bike rentals |

The regression target is:

```text
cnt
```

---

## ⚠️ Preventing Target Leakage

One of the most important preprocessing decisions in this project was removing `casual` and `registered` from the model features.

This is because:

```text
cnt = casual + registered
```

Using these two variables would therefore give the model information that directly determines the target.

For example:

```text
casual     = 100
registered = 250

cnt        = 350
```

A model trained with these variables could achieve extremely good results without actually learning how to predict future bike demand.

This is an example of **target leakage**.

Therefore, the following columns were excluded from the predictor matrix:

```python
features_to_drop = [
    "instant",
    "dteday",
    "casual",
    "registered",
    "cnt"
]
```

`cnt` was separated as the prediction target.

---

## 🔍 Exploratory Data Analysis

Before training machine-learning models, exploratory data analysis was performed to better understand the behavior of bike demand.

The analysis included:

- Distribution of hourly bike demand
- Average demand by hour
- Working-day vs non-working-day demand
- Demand under different weather conditions
- Hourly demand patterns for working and non-working days

### Hourly Demand

One of the most important relationships investigated was bike demand across different hours of the day.

Hourly demand was calculated using:

```python
hourly_demand = df.groupby("hr")["cnt"].mean()
```

This helps identify daily demand patterns and shows why the hour of the day can be an important predictor.

### Working Days

Demand was also analyzed separately for working and non-working days:

```python
hour_working = (
    df.groupby(["hr", "workingday"])["cnt"]
      .mean()
      .unstack()
)
```

This analysis demonstrates that the relationship between time and bike demand can depend on whether the observation occurs on a working day.

---

## 🛠️ Feature Engineering

Hour-of-day is a cyclical variable.

For example:

```text
23:00
00:00
```

are only one hour apart.

However, if the raw numerical values are used:

```text
|23 - 0| = 23
```

the numerical representation does not capture this cyclical relationship.

To represent time more naturally, sine and cosine features were created:

```python
X_engineered["hr_sin"] = np.sin(
    2 * np.pi * X_engineered["hr"] / 24
)

X_engineered["hr_cos"] = np.cos(
    2 * np.pi * X_engineered["hr"] / 24
)
```

This represents the hour as a position around a 24-hour cycle.

---

## ✂️ Train/Test Split

Because the observations occur over time, the dataset was sorted chronologically before splitting.

```python
df["dteday"] = pd.to_datetime(df["dteday"])

df = df.sort_values(
    ["dteday", "hr"]
).reset_index(drop=True)
```

An **80/20 chronological split** was then used.

```python
split_index = int(len(df) * 0.8)

X_train = X_engineered.iloc[:split_index]
X_test = X_engineered.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
```

The earlier 80% of observations were used for training and the later 20% for testing.

This avoids randomly mixing future observations into the training data.

---

# 🤖 Machine Learning Models

Three regression algorithms were evaluated.

## 1. Linear Regression

Linear Regression was used as the baseline model.

```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
```

Linear Regression provides a simple benchmark against which more sophisticated nonlinear models can be compared.

---

## 2. Random Forest Regression

The second model was Random Forest Regression.

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
```

Random Forest can model nonlinear relationships and interactions between variables that Linear Regression may not capture effectively.

---

## 3. Gradient Boosting Regression

The third model was Gradient Boosting Regression.

```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
```

Gradient Boosting builds trees sequentially, with later trees attempting to improve errors made by previous trees.

---

# 📏 Evaluation Metrics

Models were evaluated using three regression metrics.

### Mean Absolute Error (MAE)

MAE represents the average absolute difference between actual and predicted demand.

**Lower is better.**

### Root Mean Squared Error (RMSE)

RMSE also measures prediction error but penalizes large errors more heavily.

**Lower is better.**

### R² Score

R² measures how much of the variation in bike demand is explained by the model.

**Higher is better.**

---

# 🏆 Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Linear Regression | 122.81 | 164.35 | 0.444 |
| **Random Forest** | **44.63** | **69.13** | **0.902** |
| Gradient Boosting | 70.06 | 101.12 | 0.790 |

Random Forest produced the strongest results among the three models.

Its test-set performance was approximately:

```text
MAE  = 44.63
RMSE = 69.13
R²   = 0.902
```

The MAE indicates that the Random Forest predictions differed from actual hourly demand by approximately **45 rentals on average**.

The R² score indicates that the model explained approximately **90.2% of the variation in bike demand within the test data**.

---

## 📈 Model Comparison

The results show a large improvement when moving from a simple linear model to tree-based ensemble models.

```text
Linear Regression
        │
        │ R² = 0.444
        ▼
Gradient Boosting
        │
        │ R² = 0.790
        ▼
Random Forest
        │
        │ R² = 0.902
        ▼
   Best Model
```

The difference suggests that bike rental demand contains important **nonlinear relationships and feature interactions**.

For example, the relationship between hour and demand is unlikely to be purely linear. Demand can rise and fall throughout the day, and these patterns can also interact with variables such as working-day status and weather.

Random Forest is better suited to learning these types of relationships.

---

## 🌳 Feature Importance

Random Forest feature importance was also investigated:

```python
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)
```

This analysis helps identify which variables the trained Random Forest relied on most strongly when generating predictions.

Feature importance should not, however, be interpreted as proof that a variable **causes** changes in bike demand.

---

## 📉 Model Diagnostics

Actual-versus-predicted plots were created to visually evaluate model performance.

For a good regression model, predicted values should remain close to the diagonal:

```text
Predicted
    ↑
    |                 •
    |             •
    |          •
    |       •
    |    •
    | •
    +--------------------→ Actual
```

Residual analysis was also performed for Linear Regression to investigate patterns in prediction errors.

---

## 💡 Key Findings

The experiments produced several important observations:

1. Target leakage must be identified before training a model. Using `casual` and `registered` would make the prediction problem unrealistic.

2. Bike demand is strongly related to temporal information, making features such as hour important for the prediction problem.

3. Cyclical variables such as hour-of-day can be represented using sine and cosine transformations.

4. Tree-based ensemble models performed considerably better than the baseline Linear Regression model.

5. Random Forest achieved the strongest performance of the three tested algorithms.

6. More complex models are not automatically better. In this experiment, the tested Random Forest configuration outperformed the tested Gradient Boosting configuration.

---



## 📦 Requirements

The main Python libraries used in the project are:

```text
pandas
numpy
matplotlib
scikit-learn
datasets
jupyter
```

---

## 🔮 Possible Future Improvements

The current project intentionally focuses on establishing a complete regression workflow and comparing several standard machine-learning algorithms.

Possible future extensions include:

- Hyperparameter tuning
- Time-series cross-validation
- Additional feature engineering
- Ridge and Lasso Regression
- XGBoost or LightGBM
- More detailed residual analysis
- Permutation or SHAP-based model interpretation
- Saving and deploying the final model
- Building an interactive prediction interface

---

## 🧠 

This project demonstrates a complete introductory machine-learning regression workflow:

```text
Problem Definition
        ↓
Data Exploration
        ↓
Target Leakage Detection
        ↓
Feature Selection
        ↓
Feature Engineering
        ↓
Chronological Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Interpretation
```

The project also demonstrates why evaluating multiple models is important: a simple Linear Regression model explained about 44% of the observed variation, while Random Forest increased the test-set R² to approximately 90%.

---


