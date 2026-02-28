def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


data = pd.read_csv('Data.csv.xls')
data.fillna("Retired",inplace=True)
data["MPG"] = data["Mins"]/data['Goals']
data["xG Performance"] = data["Goals"]/data["xG"]
data.loc[data["League"].str.startswith("France Ligue"), "League"] = "Ligue 1"
top_leagues = {
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Premier League",
    "Ligue 1"
}
def weight_goals(row):
    if row["League"] in top_leagues:
        return row["Goals"] * 1
    else:
        return row["Goals"] * 0.75

data["Weighted Goals"] = data.apply(weight_goals, axis=1)

Top_scorers_total = data.sort_values(by="Goals", ascending=False).head(10)

top_weighted = data.sort_values(by="Weighted Goals", ascending=False).head(10)

top_scorers_by_year = {}

for year in data["Year"].unique():
    top_scorers_by_year[year] = (
        data[data["Year"] == year]
        .sort_values(by="Goals", ascending=False)
        .head(10)
    )

plot_data = []

for year, df in top_scorers_by_year.items():
    df = df.reset_index(drop=True)
    df["Rank"] = df.index + 1
    plot_data.append(df[["Year", "Rank", "Goals","Matches_Played"]])

plot_df = pd.concat(plot_data)

plt.figure(figsize=(10,6))
sns.lineplot(
    data=plot_df,
    x="Rank",
    y="Goals",
    hue="Year",
    marker="o"
)
plt.title("Top 10 Scorers per Year")
plt.xlabel("Rank (1 = Top Scorer)")
plt.ylabel("Goals")
plt.show()

sns.lineplot(
    data=plot_df,
    x="Rank",
    y = "Matches_Played",
    hue="Year",
    marker='o'
)
plt.title("Top 10 Scorers per Year")
plt.xlabel("Rank (1 = Top Scorer)")
plt.ylabel("Games Played")
plt.show()

sns.scatterplot(data=data,x='xG',y='xG Performance',hue='Goals',alpha=0.7,palette='coolwarm')
plt.show()

top_5_leagues = data.loc[data["League"].isin(top_leagues)]
outside_top_5 = data.loc[~data["League"].isin(top_leagues)]

from scipy.stats import ttest_ind

# Extract goals from both groups
goals_top5 = top_5_leagues["Goals"]
goals_outside = outside_top_5["Goals"]

# Perform independent t-test
t_stat, p_val = ttest_ind(goals_top5, goals_outside, equal_var=False)

print("T-statistic:", t_stat)
print("P-value:", p_val)

# Interpretation
if p_val < 0.05:
    print("Reject H0 → Players in top 5 leagues score significantly more.")
else:
    print("Fail to reject H0 → No significant difference found.")




