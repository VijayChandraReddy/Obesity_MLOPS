


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Read the data
df = pd.read_csv(r"D:\ASSIGNMENTS\kaggle\obease_data\train.csv")
X_test = pd.read_csv(r"D:\ASSIGNMENTS\kaggle\obease_test_data\test.csv")

# Define your feature columns (excluding target variable)
X = df.drop(columns=['NObeyesdad'])  # Replace 'target_column' with your actual target column name
y = df['NObeyesdad']

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Define transformers for preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numerical values with mean
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values with the most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
])

# Create the ColumnTransformer to apply different transformations to numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a pipeline that first preprocesses the data and then applies the classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))  # Set eval_metric to prevent warnings
])

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [50, 100, 200]
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate the model on the validation set
y_pred = grid_search.best_estimator_.predict(X_valid)
print(f"Accuracy on validation set: {accuracy_score(y_valid, y_pred):.2f}")






# # Remove rows with missing target, separate target from predictors
# y = X_full.NObeyesdad
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# X_full.drop(['id', 'NObeyesdad'], axis=1, inplace=True)

# # Break off validation set from training data
# X_train_full, X_valid_full, y_train, y_valid = train_test_split(
#     X_full, y_encoded, train_size=0.8, random_state=0
# )

# # Select categorical columns with relatively low cardinality
# categorical_cols = [cname for cname in X_train_full.columns if
#                     X_train_full[cname].nunique() < 10 and
#                     X_train_full[cname].dtype == "object"]

# # Select numerical columns
# numerical_cols = [cname for cname in X_train_full.columns if
#                   X_train_full[cname].dtype in ['int64', 'float64']]

# # Keep selected columns only
# my_cols = categorical_cols + numerical_cols
# X_train = X_train_full[my_cols].copy()
# X_valid = X_valid_full[my_cols].copy()

# # Preprocessing for numerical data
# numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
#                                          ('scaler', StandardScaler())])

# # Preprocessing for categorical data
# categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
#                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# # Bundle preprocessing for numerical and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[('num', numerical_transformer, numerical_cols),
#                   ('cat', categorical_transformer, categorical_cols)]
# )

# # Define hyperparameter grid for GridSearch
# param_grid = {
#     'n_estimators': [50, 100, 200],  # Number of boosting rounds
#     'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
#     'max_depth': [3, 5, 7],  # Maximum depth of a tree
# }

# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# # Perform Grid Search
# grid_search = GridSearchCV(
#     estimator=xgb, 
#     param_grid=param_grid, 
#     scoring='accuracy', 
#     cv=5, 
#     verbose=1, 
#     n_jobs=-1
# )

# # Log the grid search experiment with MLflow
# with mlflow.start_run():
#     grid_search.fit(X_train, y_train)

#     # Extract best parameters from grid search
#     best_params = grid_search.best_params_

#     # Log the best parameters
#     mlflow.log_params(best_params)
    
#     # Initialize XGBClassifier with best parameters
#     xgb_classifier = XGBClassifier(**best_params)

#     # Define pipeline
#     clf = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('model', xgb_classifier)])

#     # Train the pipeline
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_valid)

#     # Evaluate and log accuracy
#     accuracy = accuracy_score(y_valid, preds)
#     mlflow.log_metric("accuracy", accuracy)

#     print(f"Accuracy: {accuracy}")

#     # Optionally, log confusion matrix or model
#     mlflow.log_artifact("confusion_matrix.png")  # If you save a confusion matrix plot
