import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hies_state.csv")

df.head()
df.info()
df.describe()

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

#Ranks the individual states by Average Mean Income per Household
income_rank = df[['state','income_mean']] \
    .sort_values(by='income_mean', ascending=False)

##print(income_rank)

#Distributes states by income brackets, Low, Middle and High
df['income_tier'] = pd.qcut(
    df['income_mean'],
    q=3,
    labels=['Low income','Middle income','High income']
)

df[['state','income_mean','income_tier']]

#Makes comparison between the Mean and Median income, analysing Skewness
df['income_gap'] = df['income_mean'] - df['income_median']
df['gap_ratio'] = df['income_gap'] / df['income_mean']

df[['state','income_gap','gap_ratio']] \
    .sort_values(by='gap_ratio', ascending=False)

#This calculates the Severity Index in relations to the average mean income of states
df['poverty_risk'] = (
    df['poverty'] * 0.5 +
    df['gini'] * 30 +
    (1 / df['income_mean']) * 20000
)

#Higher the index value, higher the poverty
poverty_rank = df[['state','poverty_risk']] \
    .sort_values(by='poverty_risk', ascending=False)

print(poverty_rank)
