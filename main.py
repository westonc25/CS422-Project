import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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


    #One-hot for random forest
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
We can add more models to evaluate if we want. Maybe like 1-2 more.

"""

"""
Based on the model that is passed to the function along with the testing dataset this function should make predictions. 
"""
def predictions(model, X_test):

    return model.predict(X_test)

"""
Given the actual vs predicted outcomes this function will also take in the model name and the number of samples we wish 
to plot and produce visuals comparing the results. 
"""
def predictions_visuals(y_test, y_pred, model_name = "Model", n_samples = 2000):

    fig, axes = plt.subplots(1, 2, figsize = (15, 5))

    """
    A scatter plot to compare the actual vs predicted outcomes
    """
    axes[0].scatter(y_test[:n_samples], y_pred[:n_samples], alpha = 0.4, s = 10)
    # Shows whether the model over predicted or underpredicted 
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', lw = 2, label = 'Perfect Prediction')
    
    #Formatting
    axes[0].set_xlabel('Actual Volume', fontsize = 12)
    axes[0].set_ylabel('Predicted Volume', fontsize = 12)
    axes[0].set_title(f'{model_name}: Actual vs Predicted', fontsize = 14)
    axes[0].legend()
    axes[0].grid(alpha = 0.3)

    """
     A time series plot that shows how traffic volume is changing over time
     This helps us compare what happened vs. what the model predicted
    """

    # Creates an array of time points representing a 15-minute interval
    sample_indices = range(n_samples)
    
    # Plots the actual vs predicted for each time point
    axes[1].plot(sample_indices, y_test[:n_samples].values,
                 label = 'Actual', alpha = 0.7, linewidth = 1)
    axes[1].plot(sample_indices, y_pred[:n_samples],
                 label = 'Predicted', alpha = 0.7, linewidth = 1)
    
    # Formatting
    axes[1].set_xlabel('Time Index', fontsize = 12)
    axes[1].set_ylabel('Traffic Volume', fontsize = 12)
    axes[1].set_title(f'{model_name}: Time Series View', fontsize = 14)
    axes[1].legend()
    axes[1].grid(alpha = 0.3)

    #Show the results
    plt.tight_layout()

    # Save the figures generated by the function
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_predictions.png', dpi = 300)
    
    plt.show() 
    



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
    df = load_data('Automated_Traffic_Volume_Counts_20251115.csv')

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
    # Train, predict and analyze a random forest model (baseline)
    print("Now training Random forest baseline (100 trees)")
    rf_model = train_random_forest(X_train, y_train, trees=100)
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

    # Train, predict and analyze a gradient boosting model
    from sklearn.ensemble import GradientBoostingRegressor
    print("Now training Gradient Boosting model")
    gb_model = GradientBoostingRegressor(random_state=3)
    gb_model.fit(X_train, y_train)
    gb_pred = predictions(gb_model, X_test)
    print("Predictions for Gradient Boosting:", gb_pred)
    print("\n")
    gb_evaluation = evaluate_model(y_test, gb_pred, model_name="Gradient Boosting")
    print("Now evaluating results...")
    print(gb_evaluation)

    #Visualize and compare results from each model. n_samples can be set to adjust how many points are plotted
    rf_visual = predictions_visuals(y_test, rf_pred, "Random forest", n_samples = 2000)
    lr_visual = predictions_visuals(y_test, lr_pred, "Logistic regression", n_samples = 2000)
    gb_visual = predictions_visuals(y_test, gb_pred, "Gradient Boosting", n_samples=2000)

if __name__ == "__main__":
    main()