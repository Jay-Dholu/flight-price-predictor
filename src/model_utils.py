# this file contains functions for training, evaluating and saving the models

import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve


def plot_curves(sizes, mean_scores, std_scores, label, axis):
    axis.plot(sizes, mean_scores, marker='o', label=label)
    axis.fill_between(x=sizes, y1=mean_scores-std_scores, y2=mean_scores+std_scores, alpha=0.5)


def plot_learning_curves(algo_name, algo, preprocessors, x_data, y_data, figsize=(12, 4)):
    mdl = Pipeline(steps=[
        ("preprocessor", preprocessors),
        ("algorithm", algo)
    ])

    train_sizes, train_scores, test_scores = learning_curve(estimator=mdl, X=x_data, y=y_data, cv=3, scoring='r2',
                                                            n_jobs=-1, random_state=42)

    mean_train_scores = np.mean(train_scores, axis=1)
    std_train_scores = np.std(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    std_test_scores = np.std(test_scores, axis=1)

    figure, axis = plt.subplots(figsize=figsize)
    plot_curves(train_sizes, mean_train_scores, std_train_scores, "Train", axis)
    plot_curves(train_sizes, mean_test_scores, std_test_scores, "Test", axis)

    axis.set(xlabel="Training Set Sizes", ylabel="R-squared", title=algo_name)
    axis.legend(loc="lower right")
    # plt.show()              # just uncomment to see the graphs


def train_and_save_model(preprocessors, x_data, y_data, model_save_path):
    from sklearn.ensemble import RandomForestRegressor

    model = Pipeline(steps=[
        ('preprocessor', preprocessors),
        ('algorithm', RandomForestRegressor(n_estimators=10))
    ])
    model.fit(x_data, y_data)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    return model


def evaluate_model(model, x, y, metric):
    y_pred = model.predict(x)
    return metric(y, y_pred)