# Personal-Project
## About this project:
This project analyses demographic and household statistics across Malaysian states using Python. The dataset includes variables such as population size, age distribution, household structure, and urbanisation rates.

Using libraries such as pandas, matplotlib, and scikit-learn, the project performs data cleaning, correlation analysis, regression, K-Means clustering, and Principal Component Analysis (PCA). The goal is to identify patterns and relationships between demographic factors and household characteristics and visualise how states differ in their demographic profiles.

## Dataset:
The dataset was retrieved from the Department of Statistics Malaysia.

The dataset contains demographic and household statistics for Malaysian states.  
Key variables include:

- Population (thousands)
- Age distribution (0–14, 15–64, 65+)
- Total, urban, and rural households
- Average household size
- Urbanisation rate

The data was cleaned and processed using pandas before analysis.

## Analysis:
1. Correlation Analysis
2. Regression
3. K-means Clustering
4. Principle Complex Analysis

## Visualisations

The analysis generates several plots, including:

- Youth population vs household size
- Elderly population vs household size
- Urbanisation vs household size
- Cluster visualisation with regression lines
- PCA plot of Malaysian states

## Libraries
- Pandas
- Matplotlib
- Sklearn
- Numpy

## How to Run

1. Install dependencies:
pip install pandas matplotlib scikit-learn numpy

2. Place the dataset file `pop_stats.csv` in the project folder.

3. Run the script:
python analysis.py

## Future Improvements

- Add interactive visualisations
- Include more demographic variables
- Apply additional clustering evaluation methods