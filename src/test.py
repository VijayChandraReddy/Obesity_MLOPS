import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
print("tracking url scheme")
print(mlflow.get_tracking_uri())
print("\n")
