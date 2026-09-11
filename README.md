# House Price Prediction

This project predicts house sale prices using a cleaned housing dataset.
 
The work is divided into three main parts:

1. Exploratory Data Analysis
2. Data Preprocessing and Feature Engineering
3. Machine Learning Model

## Team Members and Roles

Group: BDA 2513

All three parts are combined into one pipeline in `src/main.py`. The table below
shows which part of that pipeline belongs to which student.

| Student           | Role                                       | Main Part                                             |
|-------------------|--------------------------------------------|-------------------------------------------------------|
| Makhanbet Alikhan | Dataset and EDA                            | `src/main.py`, sections 1-2; plots in `graphs/`        |
| Matmusaev Alikhan | Data Preprocessing and Feature Engineering | `src/main.py`, section 3; `data/db_preprocessed.csv`   |
| Aitzhanov Alikhan | Machine Learning Model                     | `src/main.py`, sections 4-5; results in `outputs/`     |
| Adilbek Nurdaulet | Preparation of report and DEMO             | `report/`                                              |

## Project Structure

```text
educational-practice-project-priora/
├── README.md
├── requirements.txt
├── src/
│   └── main.py
├── data/
│   ├── dataset_HousePrice.csv
│   └── db_preprocessed.csv
├── graphs/
│   ├── 01_saleprice_distribution.png
│   ├── 02_top15_correlations.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_key_features_scatter.png
│   └── 05_totalsf_vs_saleprice.png
├── outputs/
│   ├── model_metrics.csv
│   ├── validation_predictions.csv
│   └── feature_importance.csv
├── report/
│   ├── report (1).pdf
│   └── Contribution_Table.pdf
└── tests/
    └── test_structure.py
```

`data/db_preprocessed.csv`, the files in `graphs/` and the files in `outputs/`
are produced by `src/main.py`. They are kept in the repository so the results
can be checked without running the script.

## Dataset

The project uses house data with different property characteristics, such as:

- Overall quality
- Living area
- Basement area
- Garage information
- Year built
- Number of bathrooms
- Sale price

The target variable is:

```text
SalePrice
```

This is the value that the machine learning model tries to predict.

## How To Run The Project

Open the project folder in VS Code or in a terminal.

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the whole pipeline from the project root:

```bash
python src/main.py
```

The script must be started from the project root, because it reads and writes
by relative paths (`data/`, `graphs/`, `outputs/`).

It runs all three parts in order:

1. EDA, saving five plots to `graphs/`
2. Preprocessing and feature engineering, saving `data/db_preprocessed.csv`
3. Model training and evaluation, saving three files to `outputs/`

## Tests

The repository has structure tests that also check that the pipeline runs:

```bash
pytest tests/
```

## Machine Learning Approach

The machine learning part of `src/main.py` follows these steps:

1. Load the preprocessed dataset.
2. Separate features from the target column.
3. Remove the `Id` column because it is only a row identifier.
4. Split data into training and validation sets.
5. Train a `RandomForestRegressor` model.
6. Make predictions on validation data.
7. Evaluate the model using MAE, RMSE, and R2 score.
8. Show feature importance.

## Model Results

The final model is:

```text
RandomForestRegressor
```

Validation results:

| Metric | Value |
|---|---:|
| MAE | 16642.90 |
| RMSE | 24667.54 |
| R2 Score | 0.8898 |

## Prediction Examples

Prediction examples are saved in:

```text
outputs/validation_predictions.csv
```

Example:

| Actual SalePrice | Predicted SalePrice |
|---:|---:|
| 190000 | 212556.81 |
| 100000 | 97521.50 |
| 115000 | 111212.80 |

## Most Important Features

The most important features according to the model are:

1. `TotalSF`
2. `OverallQual`
3. `GrLivArea`
4. `HouseAge`
5. `BsmtFinSF1`

These features make sense because house size, quality, age, and basement area usually affect the final sale price.

## Conclusion

The project shows a complete basic machine learning workflow:

- data exploration,
- data cleaning and feature engineering,
- model training,
- prediction,
- model evaluation.

The final model gives a strong validation score for a beginner-level project and can be explained clearly during presentation.
