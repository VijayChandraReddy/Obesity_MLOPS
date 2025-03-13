# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:15:43 2024

@author: Vijay
"""
import seaborn as sns
import mlflow
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from xgboost import XGBClassifier
from category_encoders import OneHotEncoder, MEstimateEncoder
import mlflow



import dagshub
dagshub.init(repo_owner='VijayChandraReddy', repo_name='Obesity_MLOPS', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/VijayChandraReddy/Obesity_MLOPS.mlflow")
import mlflow

# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)

# Read the data
X_full = pd.read_csv(r"D:\ASSIGNMENTS\kaggle\obease_data\train.csv")
X_test = pd.read_csv(r"D:\ASSIGNMENTS\kaggle\obease_test_data\test.csv")

# Remove rows with missing target, separate target from predictors
y = X_full.NObeyesdad
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_full.drop(['id', 'NObeyesdad'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, y_encoded, train_size=0.8, random_state=0
)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if
                  X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

best_params = {'grow_policy': 'depthwise', 'n_estimators': 300, 
               'learning_rate': 0.070053726931263504, 'gamma': 0.5354391952653927, 
               'subsample': 0.5060590452456204, 'colsample_bytree': 0.38939433412123275, 
               'max_depth': 29, 'min_child_weight': 21, 'reg_lambda': 9.150224029846654e-08,
               'reg_alpha': 5.671063656994295e-08}
best_params['booster'] = 'gbtree'
best_params['objective'] = 'multi:softmax'
best_params["device"] = "cuda"
best_params["verbosity"] = 0

mlflow.set_experiment('MLOPS_MLFLOW_EXP2')
mlflow.autolog()
with mlflow.start_run():
    xgb_classifier =  XGBClassifier(**best_params)
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                     
                      ('model', xgb_classifier)
                     ])
    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_valid)

    accuracy = accuracy_score(y_valid,preds)
     
    # mlflow.log_metric('accuracy_score',accuracy)
    # mlflow.log_metric('learning_rate',best_params['learning_rate'])
    # mlflow.log_metric('max_depth',best_params['max_depth'])

    print(accuracy)
    cm = confusion_matrix(y_valid,preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm ,annot=True,fmt='d',cmap='CMRmap',)
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)

    mlflow.set_tags({'author': 'Vijay','model' : 'XGB'})
    mlflow.sklearn.log_model(xgb_classifier , 'xgb_classifier')