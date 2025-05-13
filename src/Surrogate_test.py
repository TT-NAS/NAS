import ast

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from search_algorithms import SurrogateModel
from utils import paint_results, paint_real_vs_pred


df = pd.read_csv(r"./sustituto/results_transformed_1000.csv")
seed = 402

X = df["real_codification"].apply(ast.literal_eval).apply(pd.Series)
y = df["val_iou"]

# Dividir los datos en entrenamiento y prueba
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)

paint_results(y_test, y_pred, order=True)
paint_real_vs_pred(y_test, y_pred)