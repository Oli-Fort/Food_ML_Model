import pandas as pd
import pickle

filename = '../.venv/model_food.pk1'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict([[10, 10, 10]])
print(pred)