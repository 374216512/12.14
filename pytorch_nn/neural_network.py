import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class BP(nn.Module):

    def __init__(self, data):
        super(BP, self).__init__()
        self.train_losses = []
        self.eval_losses = []
        self.fc1 = nn.Sequential(
            nn.Linear(17, 200),
            nn.PReLU(200)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.PReLU(200)
        )
        self.output = nn.Sequential(nn.Linear(200, 1))  # 输出层
        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # 制作DataLoader
        self.get_dataloader(data)

    def get_dataloader(self, data):
        data_arr = np.array(data)  # 转成ndarray类型
        self.scale = MinMaxScaler()

        data = self.scale.fit_transform(data_arr)  # 归一化
        # 制作DataLoader
        X, y = data[:, :-1], data[:, -1:]
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        train_data = zip(X_train, y_train)
        test_data = zip(X_test, y_test)
        self.train_data = DataLoader(list(train_data), 16, shuffle=True)
        self.test_data = DataLoader(list(test_data), 16, shuffle=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

    def predict(self):
        eval_loss = 0
        self.eval()  # 将模型改为预测模式
        for x_batch, y_batch in self.test_data:
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)
            output = self.forward(x_batch)
            loss = self.criterion(output, y_batch)
            eval_loss += loss.item()  # loss.item()返回的是loss的数值
        self.eval_losses.append(eval_loss / len(self.test_data))

    def train_model(self, epochs=10000):
        for epoch in range(epochs):
            train_loss = 0
            self.train()  # 将模型改为训练模式
            for x_batch, y_batch in self.train_data:
                # 前向传播
                output = self.forward(x_batch)
                loss = self.criterion(output, y_batch)
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 记录loss
                train_loss += loss.item()
            self.train_losses.append(train_loss / len(self.train_data))
            # 每完成一次迭代，就将测试集带入模型查看训练效果
            self.predict()
            print(f'epoch: {epoch}, Train Loss: {self.train_losses[-1]:.3f}, '
                  f'Test Loss: {self.eval_losses[-1]:.3f}')

    def plot_loss(self):
        x = np.arange(0, len(self.train_losses))
        plt.figure()
        plt.plot(x, self.train_losses, label='train_loss')
        plt.plot(x, self.eval_losses, label='eval_loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data = pd.read_excel('./data.xlsx').iloc[:, 1:-2]
    model = BP(data)
    model.train_model(epochs=200)
    model.plot_loss()
