import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime as dt


def load_data(file_path):
    """Load dataset from NYC Traffic volume data and preprocess it."""
    
    "Read the CSV file into the DataFrame"
    df = pd.read_csv(file_path, low_memory=False)
    
    "Convert to a readable datetime format"
    df['date_time'] = pd.to_datetime(
        df[['Yr', 'M', 'D', 'HH', 'MM']].rename(
            columns={'Yr': 'year', 'M': 'month', 'D': 'day', 'HH': 'hour', 'MM': 'minute'})
    )

    "Keep only relevant columns"
    df = df[['date_time', 'Vol', 'SegmentID', 'Boro', 'Direction', 'street', 'fromSt', 'toSt']].copy()

    # Convert Vol column to numeric, removing commas
    if df['Vol'].dtype == 'object':  
        df['Vol'] = df['Vol'].str.replace(',', '', regex=False).astype(float)
    else:
        df['Vol'] = pd.to_numeric(df['Vol'], errors='coerce')
    
    # Drop any rows where Vol couldn't be converted
    df = df.dropna(subset=['Vol'])

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

    # Rolling statistics with better NaN handling
    rolls = [4, 8, 24]  # last 1h, 2h, 6h
    for window in rolls:
        df[f'vol_roll_mean_{window}'] = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'vol_roll_std_{window}']  = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))  # Fill std NaN with 0
        df[f'vol_roll_max_{window}']  = df.groupby('SegmentID')['Vol'].transform(
            lambda x: x.rolling(window=window, min_periods=1).max())
        
    # Location encoding
    # Encodes the SegmentID with its average historical volume
    segment_mean = df.groupby('SegmentID')['Vol'].mean()
    df['segment_avg_volume'] = df['SegmentID'].map(segment_mean)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Boro', 'Direction'], drop_first=False)

    return df


def model_prep(train_df, test_df):
    # Create target: predict NEXT 15-minute interval
    train_df['target'] = train_df.groupby('SegmentID')['Vol'].shift(-1)
    test_df['target']  = test_df.groupby('SegmentID')['Vol'].shift(-1)
    
    # Drop rows where target is missing
    train_df = train_df.dropna(subset=['target'])
    test_df  = test_df.dropna(subset=['target'])
    
    # Define feature columns - FIXED: explicitly exclude string columns
    exclude = ['date_time', 'Vol', 'SegmentID', 'target', 'street', 'fromSt', 'toSt']
    features = [col for col in train_df.columns if col not in exclude]

    # Add missing columns with 0s
    for col in features:
        if col not in test_df.columns:
            test_df[col] = 0
    
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
    forest = RandomForestRegressor(
        n_estimators=trees, 
        random_state=3,
        max_depth=15,        
        min_samples_split=100,  
        min_samples_leaf=50,    
        max_features='sqrt',    
        n_jobs=-1,             
        verbose=1              
    )

    #Train the model with our datasets
    forest.fit(X_train, y_train)

    return forest


"""
Given the training datasets this function will train our model using gradient boosting regression.
"""
def train_gradient_boosting(X_train, y_train):

    gbr = GradientBoostingRegressor(
        loss='absolute_error',
        learning_rate=0.1,
        n_estimators=200,       
        max_depth=3,            
        max_features=5,
        subsample=0.8,          
        random_state=3,
        verbose=1               
    )
    gbr.fit(X_train, y_train)
    return gbr

"Given the training datasets this function will train our model using ridge regression."
def train_ridge_regression(X_train, y_train, alpha=1.0):

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    return ridge

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
Shows the model comparison and the winner based on the evaluation metrics
"""
def compare_models(evaluations):
    """Compare all models side by side"""
    comparison_df = pd.DataFrame(evaluations).T
    comparison_df = comparison_df.sort_values('mae')  # Sort by best MAE
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.to_string())
    print("\n")
    
    # Identify best model
    best_model = comparison_df['mae'].idxmin()
    print(f" Best Model (Lowest MAE): {best_model}")
    print(f" MAE: {comparison_df.loc[best_model, 'mae']:.2f} vehicles")
    print(f" R²: {comparison_df.loc[best_model, 'r2']:.4f}")
    
    return comparison_df

"""
Visualize model comparison using bar charts
"""
def plot_model_comparison(evaluations):
    """Visualize model comparison using bar charts - FIXED"""
    models = list(evaluations.keys())
    metrics = ['mae', 'rmse', 'r2']
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        values = [evaluations[model][metric] for model in models]
        axes[i].bar(models, values, color=colors)
        axes[i].set_title(f'Model Comparison: {metric.upper()}', fontsize=14)
        axes[i].set_ylabel(metric.upper(), fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()


"""
Main

"""

def main():

    """
    Begin Preprocessing the data once it is loaded in
    """

    #Stores the evaluation results for each model
    evaluations = {}

    # We first need to load the data in for preprocessing
    print("Data is being loaded!")
    df = load_data('Automated_Traffic_Volume_Counts_20251115.csv')

    print(f"Dataset size: {len(df):,} rows")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Now we can split the data in order to prepare it for training and testing
    print("Data is now being split")
    train_df, test_df = split_data_by_date(df)

    # Apply feature engineering steps in order to enhance the dataset
    train_df = create_features(train_df)
    test_df = create_features(test_df)

    # Get all columns that should be in both datasets
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Add missing columns with 0s
    for col in train_cols - test_cols:
        test_df[col] = 0
    for col in test_cols - train_cols:
        train_df[col] = 0

    # Ensure same column order
    test_df = test_df[train_df.columns]

    # Prepare data for model training
    X_train, X_test, y_train, y_test = model_prep(train_df, test_df)

    print(f"Number of features: {X_train.shape[1]}")
    print(f"Training set size: {X_train.shape[0]:,} rows")

    """
    Now that the data has been processed and prepared we can move on to conducting our experiments 
    """
    # Train, predict and analyze a random forest model (baseline)
    print("\nNow training Random forest baseline (100 trees)")
    rf_model = train_random_forest(X_train, y_train, trees=100)
    rf_pred = predictions(rf_model, X_test)
    evaluations['Random Forest'] = evaluate_model(y_test, rf_pred, model_name= "Random forest")

    # Train, predict and analyze a gradient boosting model
    print("\nNow training Gradient Boosting model")
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_pred = predictions(gb_model, X_test)
    evaluations['Gradient Boosting'] = evaluate_model(y_test, gb_pred, model_name="Gradient Boosting")

    # Train, predict and analyze a ridge regression model
    print("\nNow training Ridge regression model")
    ridge_model = train_ridge_regression(X_train, y_train)
    ridge_pred = predictions(ridge_model, X_test)
    evaluations['Ridge Regression'] = evaluate_model(y_test, ridge_pred, model_name="Ridge regression")

    # Visualize and compare results from each model. n_samples can be set to adjust how many points are plotted
    print("\nGenerating visualizations...")
    rf_visual = predictions_visuals(y_test, rf_pred, "Random forest", n_samples = 2000)
    gb_visual = predictions_visuals(y_test, gb_pred, "Gradient Boosting", n_samples=2000)
    ridge_visual = predictions_visuals(y_test, ridge_pred, "Ridge regression", n_samples=2000)

    # Visualized model comparison
    plot_model_comparison(evaluations)

    # Compare all models
    compare_models(evaluations)

if __name__ == "__main__":
    main()