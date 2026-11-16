import joblib

bundle = joblib.load("datasets/model.pkl")
print(bundle.keys())  # dict_keys(['model', 'columns'])
print(bundle["model"])  # XGBClassifier(...)
print(bundle["columns"][:5])  # first 5 feature names
