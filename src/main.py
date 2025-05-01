import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from data_utils import load_data, split_data
from preprocessors import build_preprocessor
from model_utils import plot_learning_curves, train_and_save_model, evaluate_model


def main():
    # Settings
    pd.set_option("display.max_columns", None)

    # Load Data
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_df, val_df, test_df = load_data(
        data_dir / 'training.csv',
        data_dir / 'validation.csv',
        data_dir / 'testing.csv'
    )

    x_train, y_train = split_data(train_df)
    x_val, y_val = split_data(val_df)
    x_test, y_test = split_data(test_df)

    # Build Preprocessor
    preprocessors = build_preprocessor()

    # Combine Train + Val
    data = pd.concat([train_df, val_df], axis=0)
    x_data, y_data = split_data(data)

    # Algorithms
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Support Vector Machine": SVR(),
        "Random Forest": RandomForestRegressor(n_estimators=10),
        "XG Boost": XGBRegressor(n_estimators=10)
    }

    for algorithm_name, algorithm in algorithms.items():
        plot_learning_curves(algorithm_name, algorithm, preprocessors, x_data, y_data)

    # Train final model and saving it
    model_save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'flight-price-predictor.joblib')
    model = train_and_save_model(preprocessors, x_data, y_data, model_save_path)

    # Evaluate
    print(f"Training R2 Score: {evaluate_model(model, x_data, y_data, r2_score):.4f}")
    print(f"Test R2 Score: {evaluate_model(model, x_test, y_test, r2_score):.4f}")


if __name__ == '__main__':
    main()
