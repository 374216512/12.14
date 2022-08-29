from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import pickle

data = pd.read_excel('DNN数据.xlsx').iloc[1:]
features = data.iloc[:, 1:4].values  # 36 * 3
actual_value = data.iloc[:, 4:].values  # 36 * 3

epochs = 30000
batch_size = 10

hidden_layer_nodes = (200, 300)

mlp_reg = MLPRegressor(hidden_layer_nodes, verbose=True, max_iter=epochs,
                       batch_size=batch_size, n_iter_no_change=20, learning_rate_init=0.001,
                       tol=0.0001)

mlp_reg.fit(features, actual_value)

with open('model.pickle', 'wb') as f:
    pickle.dump(mlp_reg, f)

pred = mlp_reg.predict(np.array([100, 8.0, 1.5]).reshape(1, -1))
print(pred)
