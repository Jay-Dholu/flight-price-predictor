# this file contains pipelines for preprocessing

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engine.datetime import DatetimeFeatures


def build_preprocessor():
    datetime_columns = ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']
    numeric_columns = ['Duration', 'Total_Stops']
    categorical_columns = ['Airline', 'Source', 'Destination', 'Additional_Info']

    doj_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("extractor", DatetimeFeatures(features_to_extract=['week', 'day_of_week', 'month', 'day_of_month'], format='mixed')),
        ("scaler", StandardScaler())
    ])

    time_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("extractor", DatetimeFeatures(features_to_extract=['hour', 'minute'], format='mixed')),
        ("scaler", StandardScaler())
    ])

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numeric_columns),
        ('categorical', categorical_transformer, categorical_columns),
        ('doj', doj_transformer, ['Date_of_Journey']),
        ('time', time_transformer, ['Dep_Time', 'Arrival_Time'])
    ])

    return preprocessor
