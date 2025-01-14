
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
import os
import shutil
import csv



lr = 0.0005
batch_size = 512
epochs = 300

df = pd.read_csv("input.csv", delimiter=',', encoding='utf-8')

def remove_outliers(df, feature):
    df = df[df[feature] != -1]
    Q1 = df[feature].quantile(0.2)
    Q3 = df[feature].quantile(0.8)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

# 删除含有NaN值的行
df = df.dropna()
df = df.iloc[:,1:]
df = df.astype('float64')
# 删除任何包含小于0的值的行
df = df[(df>= 0.).all(axis=1)]#& (df.iloc[:, 2] <= 1000)

print('shaixuan',len(df))
# 再次检查是否还有NaN值
#print(df.isnull().sum())
import numpy as np


df.iloc[:, 0] = np.log(df.iloc[:, 0])  # 对第一列进行对数转换     Max_IDR
df.iloc[:, 1] = np.log(df.iloc[:, 1])  # 对第二列进行对数转换     Max_PFA
df.iloc[:, 2] = np.log(df.iloc[:, 2])  # 对第三列进行对数转换     Top_PFA
df.iloc[:, 3] = np.log(df.iloc[:, 3])  # 对第四列进行对数转换     top_IDR
df.iloc[:, 4] = np.log(df.iloc[:, 4])  # 对输出列进行对数转换     nStory
df.iloc[:, 5] = np.log(df.iloc[:, 5])  # 对输出列进行对数转换     storyheight
df.iloc[:, 6] = np.log(df.iloc[:, 6])  # 对输出列进行对数转换     year
df.iloc[:, 7] = np.log(df.iloc[:, 7])  # 对输出列进行对数转换     strutype
df.iloc[:, 8] = np.log(df.iloc[:, 8])  # 对输出列进行对数转换     Earthquake Magnitude
df.iloc[:, 9] = np.log(df.iloc[:, 9])  # 对输出列进行对数转换     EpiD (km)
df.iloc[:, 10] = np.log(df.iloc[:, 10])  # 对输出列进行对数转换   Vs30
df.iloc[:, 11] = np.log(df.iloc[:, 11])  # 对输出列进行对数转换   PGA
df.iloc[:, 12] = np.log(df.iloc[:, 12])  # 对输出列进行对数转换   Significant_Duration
df.iloc[:, 13] = np.log(df.iloc[:, 13])  # 对输出列进行对数转换   Arias_Intensity
df.iloc[:, 14] = np.log(df.iloc[:, 14])  # 对输出列进行对数转换   Cumulative_Absolute_Velocity
df.iloc[:, 15] = np.log(df.iloc[:, 15])  # 对输出列进行对数转换   Peak_Ground_Velocity (PGV)


cleaned_df = df.copy()
cleaned_df = cleaned_df.replace([float('inf'), float('-inf')],  np.nan).astype(float)
cleaned_df = cleaned_df.dropna()
print("原始数据行数:", len(df))
print("清理后的数据行数:", len(cleaned_df))
print(cleaned_df.isnull().sum())

current_time = datetime.now().strftime("%Y%m%d%H%M%S")

X = cleaned_df.iloc[:, 4:16]
y = cleaned_df.iloc[:,2]


print(X)
print(y)
# 检查并清理NaN值
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

torch.manual_seed(1)
np.random.seed(1)
filtered_data = pd.DataFrame(X)
filtered_data['Output'] = y  # 将处理后的输出列添加到数据框中

filtered_data.to_csv(os.path.join(folder_path,'filtered_dataPFA.csv'), index=False)

X_train, X_all, y_train, y_all = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_all,y_all,test_size=0.5,random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

print(X_train.shape[1])
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
        nn.Linear(X_train.shape[1], 512),
        nn.LeakyReLU(0.01),  # 0.01
        nn.BatchNorm1d(512),
        nn.Dropout(0),

        nn.Linear(512, 512),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(512),
        nn.Dropout(0),

        nn.Linear(512, 512),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(512),
        nn.Dropout(0),

        nn.Linear(512, 512),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(512),
        nn.Dropout(0),

        nn.Linear(512, 256),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(256),
        nn.Dropout(0),

        nn.Linear(256, 256),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(256),
        nn.Dropout(0),

        nn.Linear(256, 128),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(128),
        nn.Dropout(0),

        nn.Linear(128, 64),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(64),
        nn.Dropout(0),

        nn.Linear(64, 32),
        nn.LeakyReLU(0.01),  # LeakyReLU(0.01)
        nn.BatchNorm1d(32),
        nn.Dropout(0),

        nn.Tanh(),
        nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = DNN().to(device)

criterion = nn.MSELoss()   #损失函数选择
optimizer = optim.Adam(model.parameters(), lr=lr)    #优化算法选择

# 训练模型
def train_model(epochs):
    train_losses = []
    test_losses = []
    best_loss = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                test_loss += loss.item()

        test_losses.append(test_loss / len(test_loader))

        if test_losses[epoch] < best_loss:
            best_loss = test_losses[epoch]
            save_path = 'my_best_model.pth'
            torch.save(model.state_dict(), save_path)
        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Test Loss: {test_loss / len(test_loader)}')

    file_path3 = 'loss.txt'
    with open(file_path3, 'w') as file:
        file.write('train_loss' + '\t' + 'test_loss' + '\n')
        for i in range(len(train_losses)):
            file.write(str(train_losses[i]) + '\t' + str(test_losses[i]) + '\n')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"loss_{current_time}.png")
    plt.show()
    return model

trained_model = train_model(epochs)

trained_model = DNN().to(device)
def validate_model(model, test_loader):
    model.load_state_dict(torch.load('my_best_model.pth'))
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.extend(output.view(-1).tolist())
            actuals.extend(y_batch.view(-1).tolist())

    r2 = r2_score(actuals, predictions)
    # 计算 MAE
    mae = mean_absolute_error(actuals, predictions)
    print(f'Mean Absolute Error (MAE): {mae}')

    # 计算 MSE
    mse = mean_squared_error(actuals, predictions)
    print(f'Mean Squared Error (MSE): {mse}')

    # 计算 RMSE
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    # 绘制预测值和实测值的散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r')  # 45 degree line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'R^2 Score: {r2_score(actuals, predictions):.4f}')

    plt.savefig(f"Predicted_vs_actual_{current_time}.png")
    plt.show()

    # 指定txt文件路径
    file_path = 'true_pre.txt'
    # 打开文件以写入数据
    with open(file_path, 'w') as file:
        file.write('true' + '\t' + 'Predictions' + '\n')
        for i in range(len(actuals)):
            file.write(str(actuals[i]) + '\t' + str(predictions[i]) + '\n')
    file_path2 = '指标.txt'
    with open(file_path2, 'w') as file:
        file.write('mae =' + str(mae) + '\n' +'mse =' + str(mse) + '\n' + 'rmse =' + str(rmse))


validate_model(trained_model, val_loader)