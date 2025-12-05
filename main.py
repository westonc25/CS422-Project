import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime as dt


def load_data(file_path):
    """Load dataset from NYC Traffic volume data and preprocess it."""
    
    "Read the CSV file into the DataFrame"
    df = pd.read_csv(file_path)
    
    "Convert to a readable datetime format"
    df['date_time'] = pd.to_datetime(
        df[['Yr', 'M', 'D', 'HH', 'MM']].rename(
            columns={'Yr': 'year', 'M': 'month', 'D': 'day', 'HH': 'hour', 'MM': 'minute'})
    )

    "Keep only relevant columns"
    df = df[['date_time', 'Vol', 'SegmentID', 'Boro', 'Direction', 'street', 'fromSt', 'toSt']].copy()

    "Sort chronologically"
    df = df.sort_values(['SegmentID', 'Direction', 'date_time']).reset_index(drop=True)
    
    return df


def split_data_by_date(df):

    "The cutoff date for splitting the dataset"
    cutoff_date = pd.to_datetime('2024-09-01')

    "Logic for splitting the dataset based on the cutoff date"
    train_df = df[df['date_time'] < cutoff_date].copy()
    test_df = df[df['date_time'] >= cutoff_date].copy()

    return train_df, test_df


def create_features(df):
    """Several feature engineering steps to enhance the dataset."""
    
    # Time features
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek      # 0 = Monday
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['dayofweek'] >= 5
    df['is_rush_hour'] = df['hour'].isin([7,8,9,16,17,18]) & (~df['is_weekend'])

    # Rolling statistics
    rolls = [4, 8, 24]  # last 1h, 2h, 6h
    for window in rolls:
        df[f'vol_roll_mean_{window}'] = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'vol_roll_std_{window}']  = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std())
        df[f'vol_roll_max_{window}']  = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max())
        
    # Location encoding
    # Ecodes the SegmentID with its average historical volume
    segment_mean = df.groupby('SegmentID')['Vol'].mean()
    df['segment_avg_volume'] = df['SegmentID'].map(segment_mean)


    # One-hot encode categorical location/direction
    df = pd.get_dummies(df, columns=['Boro', 'Direction'], drop_first=True)

    return df


def model_prep(train_df, test_df):
    # Create target: predict NEXT 15-minute interval
    train_df['target'] = train_df.groupby('SegmentID')['Vol'].shift(-1)
    test_df['target']  = test_df.groupby('SegmentID')['Vol'].shift(-1)
    
    # Drop rows where target is missing
    train_df = train_df.dropna(subset=['target'])
    test_df  = test_df.dropna(subset=['target'])
    
    # Define feature columns
    exclude = ['date_time', 'Vol', 'SegmentID', 'target']
    features = [col for col in train_df.columns if col not in exclude]
    
    X_train = train_df[features]
    X_test  = test_df[features]
    y_train = train_df['target']
    y_test  = test_df['target']
    
    print(f"Final dataset ready!")
    print(f"   → Features: {len(features)}")
    print(f"   → Train samples: {len(X_train):,}")
    print(f"   → Test samples : {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test

"""
Given the training datasets and a value for how many trees we want to construct
the function will train our model using random forest regression. 

"""
def train_random_forest(X_train, y_train, trees):

    #Load the training function
    forest = RandomForestRegressor(n_estimators = trees, random_state = 3)

    #Train the model with our datasets
    forest.fit(X_train, y_train)

    return forest

"""
Given the training datasets and a value for the C parameter 
the function will train our model using random forest regression 

Different values of C allow us to analayze how regularization is taking place

"""
def train_logistic_regression(X_train, y_train, c_parameter):

    #Load the training function
    logreg = LogisticRegression(c = c_parameter)

    #Train the model with our datasets
    logreg.fit(X_train, y_train)

    return logreg

"""
Given the training datasets this function will train our model using gradient boosting regression.
"""
def train_gradient_boosting(X_train, y_train):

    gbr = GradientBoostingRegressor(loss='absolute_error',
                                learning_rate=0.1,
                                n_estimators=300,
                                max_depth = 1, 
                                max_features = 5)
    gbr.fit(X_train, y_train)
    return gbr

"""
Based on the model that is passed to the function along with the testing dataset this function should make predictions. 
"""
def predictions(model, X_test):

    return model.predict(X_test)


def grid_search_random_forest(X_train, y_train):
    """Grid-search for RandomForestRegressor with a small search grid."""
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor(random_state=3)
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"RandomForest best params: {grid.best_params_}")
    print(f"RandomForest best CV MAE: {-grid.best_score_:.2f}")

    return grid.best_estimator_


def grid_search_gradient_boosting(X_train, y_train):
    """Grid-search for GradientBoostingRegressor with a small search grid."""
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    gb = GradientBoostingRegressor(random_state=3)
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(gb, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"GradientBoosting best params: {grid.best_params_}")
    print(f"GradientBoosting best CV MAE: {-grid.best_score_:.2f}")

    return grid.best_estimator_


def cross_val_evaluate(model, X_train, y_train, model_name="Model"):
    """Evaluate model using TimeSeriesSplit cross-validation and report MAE per fold."""
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae_scores = -scores

    print(f"\n{model_name} Cross-Validation Results:")
    print(f" MAE per fold: {[f'{s:.2f}' for s in mae_scores]}")
    print(f" Mean MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")

    return mae_scores

def evaluate_model(y_test, y_pred, model_name = "Model"):

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred) 

    print(f"\n{model_name} Performance:")
    print(f" MAE: {mae:.2f} vehicles")
    print(f" RMSE: {rmse:.2f} vehicles")
    print(f" R2: {r2:.4f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


"""
Main

"""

def main():

    """
    Begin Preprocessing the data once it is loaded in
    """
    # We first need to load the data in for preprocessing
    print("Data is being loaded!")
    df = load_data()

    # Now we can split the data in order to prepare it for training and testing
    print("Data is now being split")
    train_df, test_df = split_data_by_date(df)

    # Apply feature engineering steps in order to enhance the dataset
    train_df = create_features(train_df)
    test_df = create_features(test_df)

    # Prepare data for model training
    X_train, X_test, y_train, y_test = model_prep(train_df, test_df)

    """
    Now that the data has been processed and prepared we can move on to conducting our experiments 
    """
    # Train, predict and analyze a random forest model
    print("Now training Random forest model")
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = predictions(rf_model, X_test)
    print("Predictions for Random Forest:", rf_pred)
    print("\n")
    rf_evaluation = evaluate_model(y_test, rf_pred, model_name= "Random forest")
    print("Now evaluating results...")
    print(rf_evaluation)

    

    # Train, predict and analyze a logistic regression model
    print("Now training Logistic regression model")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_pred = predictions(lr_model, X_test)
    print("Predictions for Logistic regression:", lr_pred)
    print("\n")
    lr_evaluation = evaluate_model(y_test, lr_pred, model_name= "Logistic regression")
    print("Now evaluating results...")
    print(lr_evaluation)

    
