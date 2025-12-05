# CS422-Project

## Traffic Volume Prediction Project

## Dataset
- **File**: Automated_Traffic_Volume_Counts_20251115.csv
- **Split**: Training data before Sept 1, 2024; test data Sept 1 onward
- **Source**: https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/data_preview

## Features
- **Time Based**: 
    - Hour of Day for date_time
    - Day of Week for date_time
    - Month for date_time
    - Is Weekend
    - Is Rush Hour
- **Rolling Statistics**: 1-hour, 2-hour, and 6-hour rolling means, std, and max volumes
- **Location Encoding**: average historical volume per segment, one-hot encoded borough and direction



## Models Compared

### Random Forest
- Primary Model
- Baseline of 100 trees w/ `random state=3`

### Gradient Boosting
- Default sklearn GradientBoostingRegressor

### Ridge Regression
- Default sklearn Ridge regression

### Evaluation Metrics
- **MAE**: Mean Absolute Error (vehicles)
- **RMSE**: Root Mean Squared Error (vehicles)
- **R²**: Coefficient of determination

## Files
- `main.py`: Main pipeline (data loading, preprocessing, feature engineering, model training, evaluation)
- `Automated_Traffic_Volume_Counts_20251115.csv`: Input data
- `README.md`: This file



## How to Run

### Prerequisites
Install required packages:
```
pip install pandas numpy scikit-learn matplotlib
```

### Execute the Pipeline
```
cd ".\CS422-Project"
python main.py
```


## Output

The program will:

1. Load and process the data
2. Created tailored features for the problem
3. Train a random forest, gradient boosting and ridge regression model
4. Prints the evaluation metrics to the terminal
5. Generates visualization for all of the models
6. Show a comparison of all the models in the terminal and shows which model is the best

