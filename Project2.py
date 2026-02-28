import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

pd.set_option('display.max_columns', None)

# Load and filter advanced stats
advanced = pd.read_csv('advanced.csv').sort_values(by="PER", ascending=False)
advanced_filtered = advanced.loc[advanced["MP"] > 1000].drop(columns=["Player-additional", "Awards", "Rk"])

# Load and clean salary data
salaries = pd.read_csv('salaries_new.csv')
salaries_2025 = (
    salaries.iloc[1:, :3]
            .reset_index(drop=True)
            .rename(columns={"Unnamed: 0": "Player", "Unnamed: 1": "Team", "Salary": "Salary"})
)
salaries_2025.drop("Team", axis=1, inplace=True)
salaries_2025["Salary"] = salaries_2025["Salary"].replace({r'\$': '', ',': ''}, regex=True).astype(float)

# Merge datasets
data = advanced_filtered.merge(salaries_2025, how="left", on="Player").drop_duplicates(subset="Player")
data = data.dropna(subset=["Salary"])

print(data.shape)

# Select numeric features
X = data.drop(["Salary"], axis=1).select_dtypes(include=[np.number])
y = data["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42018)

# ----------------------------
# Linear Regression (unscaled)
# ----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Unscaled training R²:", lr.score(X_test, y_test))

# ----------------------------
# Linear Regression (scaled)
# ----------------------------
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

lr.fit(X_scaled_train, y_train)
print("Scaled training R²:", lr.score(X_scaled_test, y_test))

# ----------------------------
# Lasso and Ridge (scaled)
# ----------------------------
ls = Lasso(alpha=10, max_iter=50000)  # increased alpha & iterations to converge
rr = Ridge(alpha=1, max_iter=10000)

ls.fit(X_scaled_train, y_train)
rr.fit(X_scaled_train, y_train)

print("Lasso training R²:", ls.score(X_scaled_test, y_test))
print("Ridge training R²:", rr.score(X_scaled_test, y_test))

# ----------------------------
# Polynomial Ridge Pipeline
# ----------------------------
pipeline = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1))
])
pipeline.fit(X_train, y_train)
print("Polynomial pipeline R²:", pipeline.score(X_test, y_test))

# Grid search for best polynomial degree and alpha
param_grid = {
    "polynomial__degree": [1, 2, 3, 4],
    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]
}

search = GridSearchCV(pipeline, param_grid, n_jobs=2)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best pipeline R²:", best.score(X_test, y_test))


def get_R2_features(model, test=True):
    # X: global
    features = list(X)


    R_2_train = []
    R_2_test = []

    for feature in features:
        model.fit(X_train[[feature]], y_train)

        R_2_test.append(model.score(X_test[[feature]], y_test))
        R_2_train.append(model.score(X_train[[feature]], y_train))

    plt.bar(features, R_2_train, label="Train")
    plt.bar(features, R_2_test, label="Test")
    plt.xticks(rotation=90)
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    print(
        "Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)), str(np.mean(R_2_test))))
    print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)), str(np.max(R_2_test))))



# Access the Ridge model inside the pipeline
ridge_model = best.named_steps["model"]

# Get coefficients
coefficients = ridge_model.coef_
intercept = ridge_model.intercept_

print("Intercept:", intercept)
print("Coefficients:", coefficients)

feature_names = best.named_steps["polynomial"].get_feature_names_out(X.columns)
for name, coef in zip(feature_names, coefficients):
    print(name, coef)

# ----------------------------
# Plot Top 10 Coefficients
# ----------------------------
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Use absolute value to rank importance
coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
top10 = coef_df.sort_values(by="Abs_Coefficient", ascending=False).head(10)


plt.figure(figsize=(10, 6))

# Sort so largest appears at top visually
top10_sorted = top10.sort_values(by="Coefficient")

plt.barh(top10_sorted["Feature"], top10_sorted["Coefficient"])

plt.xlabel("Coefficient Value (USD)")
plt.title("Top 10 Most Influential Polynomial Ridge Coefficients")

# Format x-axis in millions for readability
def millions(x, pos):
    return f"${x/1_000_000:.1f}M"

plt.gca().xaxis.set_major_formatter(FuncFormatter(millions))

plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

