Machine Learning Project 1
# Adult Income Classification – Project1

A notebook that predicts whether a person's annual income exceeds \$50K using the UCI Adult dataset format. It covers data cleaning, feature engineering, classic linear models (Perceptron, Adaline), and strong baselines (Logistic Regression and SVM), with clear evaluation outputs.

## Repository Structure

```
.
├── Project1.ipynb
├── data/
│   ├── project_adult.csv
│   └── project_validation_inputs.csv
├── Group_19_Perceptron_PredictedOutputs.csv
├── Group_19_Adaline_PredictedOutputs.csv
├── Group_19_LogisticRegression_PredictedOutputs.csv
├── Group_19_SVM_PredictedOutputs.csv
└── README.md
```

> The notebook expects the following CSVs by default: **project_adult.csv, project_validation_inputs.csv**. Place them in `data/` or adjust the paths at the top of the notebook.

## What’s Inside

Notebook sections:
- Data ingestion and cleaning
- Handeling missing values and Scaling data
- Perceptron Model
- Adaline Model
- Logistic Regression
- Logistic Regression (CV)
- SVM (SVC)
- Generating ouptput csv's

Key preprocessing steps:
- Replace `'?'` with `NaN`, then impute missing values **with the mode** per column.
- One-hot encode categorical features with `drop_first=True` to avoid dummy traps.
- Standardize numeric columns using `StandardScaler` (fit on train, transform test).
- Train/validation split with **TEST_SIZE=0.20**, **RANDOM_SEED=35** (stratified by target).

### Feature Schema

- **Target:** `income` mapped to `1` for `">50K"` and `0` for `"<=50K"`.
- **Categorical columns:** workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Numerical columns:** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

If your input CSV uses slightly different names (e.g., extra spaces or casing), the notebook normalizes column names by stripping, lowercasing, and replacing spaces with dashes.


1) **Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn
```

2) **Data**
- Put your CSVs in `./data/` (or adjust the read paths in the notebook).

3) **Run**
```bash
jupyter lab  # or: jupyter notebook
# open Project1.ipynb and Run All
```

## Outputs & Evaluation

The notebook prints standard classification metrics for the logistic and SVM models using scikit-learn’s `classification_report`, and visualizes confusion matrices for selected models. It also compiles a `results_df` table of model performance.


## Using the Validation Inputs

The notebook loads `project_validation_inputs.csv` and applies the same cleaning/imputation steps. To generate predictions with your best model:

There are 4 output files in the repository with the best performing models of each type. 

To produce an output csv with predictions for the validation set un comment the code in the last cell and enter whatever model you would like.




