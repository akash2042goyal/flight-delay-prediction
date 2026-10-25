# ✈️ Flight Delay Prediction

A machine learning project that analyses **2019 US airline departure delays** and builds models to (1) predict whether a flight will be delayed by 15+ minutes and (2) estimate airport passenger volume. It covers the full workflow: data cleaning, exploratory analysis, model training and comparison, and an interactive HTML dashboard that summarises the results.

## Table of contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Dashboard](#dashboard)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [Author](#author)

## Overview

Flight delays cost airlines and passengers time and money. This project asks:

- **Can we predict whether a departure will be delayed by 15 minutes or more?** (binary classification)
- **Which factors (month, time of day, carrier, airport, weather, plane age, congestion) are associated with delays?** (exploratory analysis)
- **Can airport passenger volume be modelled from the other features?** (regression)

## Dataset

**Source:** [2019 Airline Delays and Cancellations (Kaggle)](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations), file `full_data_flightdelay.csv`.

The data is **not included** in this repository because of its size. Download it from Kaggle (see [Getting started](#getting-started)).

**Columns used in the analysis**

| Column | Description |
|---|---|
| `DEP_DEL15` | **Target.** 1 if the departure was delayed 15+ minutes, otherwise 0 |
| `MONTH` | Month of the flight |
| `DEP_TIME_BLK` | Scheduled departure time block |
| `CARRIER_NAME` | Airline |
| `DEPARTING_AIRPORT` | Origin airport |
| `PREVIOUS_AIRPORT` | Airport of the aircraft's previous leg (`NONE` if not available) |
| `PLANE_AGE` | Age of the aircraft |
| `CONCURRENT_FLIGHTS` | Number of flights scheduled at the same airport and time |
| `TMAX`, `PRCP`, `AWND` | Weather: maximum temperature, precipitation, average wind speed |
| `AVG_MONTHLY_PASS_AIRPORT` | Average monthly passengers at the airport (regression target) |

## Methodology

1. **Loading and sampling:** the dataset has millions of rows, so the script takes a **stratified sample of up to 500,000 rows**, which keeps the delayed/on-time ratio the same as the full data (`random_state=42` for reproducibility).
2. **Cleaning:** missing wind-speed (`AWND`) values are filled with the median; duplicate rows are checked and missing values reported.
3. **Encoding:** categorical columns (carrier, departure time block, airports) are label-encoded for correlation analysis and one-hot encoded for modelling. A `HAS_PREV_AIRPORT` flag replaces the high-cardinality previous-airport column.
4. **Scaling:** features are standardised with `StandardScaler`, fitted on the training split only to avoid data leakage.
5. **Train/test split:** 80/20, stratified on the target for classification.
6. **Exploratory analysis:** correlation heatmap and Matplotlib and Seaborn charts (line, bar, histogram, scatter, pie, box and pair plots), including delay rate by month and the relationship between weather and delays.

## Models

### Classification: will the flight be delayed 15+ minutes?

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=500`, `class_weight='balanced'` |
| Decision Tree | `max_depth=8`, `class_weight='balanced'` |
| K-Nearest Neighbours | `n_neighbors=7`, trained on a 50,000-row sample (KNN scales poorly with large training sets) |

Delayed flights are the minority class, so `class_weight='balanced'` is used to stop the models from simply predicting "on time" for everything. Models are compared on **accuracy, precision, recall and F1-score**, with side-by-side **confusion matrices**. The best model is chosen by **F1-score**, which balances precision and recall on imbalanced data.

### Regression: airport passenger volume

| Model | Settings |
|---|---|
| Linear Regression | All features |
| Polynomial Regression | Degree 2 on the 5 features most correlated with the target |

Compared on **MAE, RMSE and R²**. The best model is chosen by the lowest RMSE.

<!--
### Results (fill in after running the script)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Decision Tree | | | | |
| KNN | | | | |
-->

## Dashboard

The last section of the script generates `flight_delay_website.html`, a self-contained dashboard (no server needed) with six sections:

1. **Overview:** headline numbers and the best models
2. **Patterns:** delays by month, carrier, time of day, airport and weather
3. **Statistics:** descriptive statistics for every numeric column
4. **Classification:** model comparison and confusion matrices
5. **Regression:** linear vs polynomial results
6. **Insights:** key takeaways written automatically from the results

In Google Colab it shows an **Open in new tab** button; when run locally it opens in your default browser.

## Getting started

**Requirements:** Python 3.9+ and the packages in `requirements.txt` (pandas, NumPy, Matplotlib, Seaborn, scikit-learn).

### Run locally (VS Code, terminal, any editor)

```bash
# 1. Clone the repository
git clone https://github.com/akash2042goyal/flight-delay-prediction.git
cd flight-delay-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
mkdir data
# Download full_data_flightdelay.csv (or the .csv.zip) from Kaggle into data/

# 4. Run
python flight_delay_analysis.py
```

You can also pass the dataset path directly:

```bash
python flight_delay_analysis.py path/to/full_data_flightdelay.csv
```

> Charts open one window at a time. Close each window to move on to the next step. The dashboard opens in your browser at the end.

### Run in Google Colab

1. Create a new notebook and paste in the contents of `flight_delay_analysis.py`.
2. Upload the dataset to `/content/`.
3. Run the cell.

## Project structure

```
flight-delay-prediction/
├── flight_delay_analysis.py   # preprocessing, EDA, models, dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # project documentation
├── LICENSE                    # MIT license
├── .gitignore                 # keeps large data files out of the repo
└── data/                      # (create locally) place the Kaggle dataset here
```

Running the script also creates `flightdelay_preprocessed.csv` and `flight_delay_website.html` in the working folder.

## Limitations

- **Sampling:** results come from a 500,000-row sample, and KNN is trained on 50,000 rows, so metrics can differ slightly from a full-data run.
- **Passenger regression:** `AVG_MONTHLY_PASS_AIRPORT` is mostly fixed per airport, so that model describes airport size rather than delays. It is a supporting analysis, not a delay predictor.
- **Weak individual correlations:** no single feature strongly predicts delays. They depend on a combination of factors, so classification accuracy has a natural ceiling with these features.
- **2019 only:** the model reflects 2019 patterns and may not generalise to other years.

## Future improvements

- Try ensemble models (Random Forest, Gradient Boosting) and tune hyperparameters with cross-validation
- Add feature importance and model explainability (for example SHAP)
- Train on the full dataset or add more years of data
- Deploy the dashboard as a hosted web app

## Author

**Akash Goyal** · GitHub: [@akash2042goyal](https://github.com/akash2042goyal)

Dataset credit: [threnjen on Kaggle](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations).
