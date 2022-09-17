import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader


class BP(nn.Module):
    def __init__(self, data, lr=1e-3):
        super(BP, self).__init__()
        self.train_data = None
        self.test_data = None
        self.train_losses = []
        self.test_losses = []
        self.fc1 = nn.Sequential(
            nn.Linear(17, 200),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(200, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.get_dataloader(data)

    def get_dataloader(self, data):
        data_arr = np.array(data)
        self.scaler = MinMaxScaler()
        data_arr = self.scaler.fit_transform(data_arr)  # 进行归一化处理
        X, y = data_arr[:, :-1], data_arr[:, -1:]
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        train_data = zip(x_train, y_train)
        test_data = zip(x_test, y_test)
        self.train_data = DataLoader(list(train_data), batch_size=16, shuffle=True)
        self.test_data = DataLoader(list(test_data), batch_size=16, shuffle=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

    def predict(self):
        test_loss = 0
        self.eval()  # 切换到测试模式
        for x_batch, y_batch in self.test_data:
            x_batch = torch.autograd.Variable(x_batch)
            y_batch = torch.autograd.Variable(y_batch)
            output = self.forward(x_batch)
            loss = self.criterion(output, y_batch)
            test_loss += loss.item()
        self.test_losses.append(test_loss / len(self.test_data))

    def train_model(self, epochs=300):
        for epoch in range(epochs):
            train_loss = 0
            self.train()  # 切换到训练模式
            for x_batch, y_batch in self.train_data:
                output = self.forward(x_batch)
                loss = self.criterion(output, y_batch)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.train_losses.append(train_loss / len(self.train_data))
            self.predict()
            print(f'epoch: {epoch}, Train Loss: {self.train_losses[-1]:.5f}, '
                  f'Test Loss: {self.test_losses[-1]:.5f}')

    def plot_loss(self):
        x = np.arange(0, len(self.train_losses))
        plt.plot(x, self.train_losses, label='train loss')
        plt.plot(x, self.test_losses, label='test loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    df = pd.read_excel('./data.xlsx').iloc[:, 1:-2]
    net = BP(df)
    net.train_model()
    net.plot_loss()
