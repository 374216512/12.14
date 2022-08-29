import pickle
import numpy as np

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(np.array([
    [100, 8.0, 1.5],
    [100, 8.0, 2.5]
]).reshape(2, -1))
print(pred)
