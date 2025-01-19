import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
import os
import shutil
import csv
# Set the learning rate, batch_size, and number of training epochs.
lr = 0.0005
batch_size = 512
epochs = 4
# Create the "model" folder if it does not exist.
if not os.path.exists('model'):
    os.makedirs('model')
df = pd.read_csv("全部数据-input.csv", delimiter=',', encoding='utf-8')
# Delete rows that contain NaN values.
df = df.dropna()
df = df.iloc[:, 1:]
df = df.astype('float64')
# Delete any rows that contain values less than zero
df = df[(df >= 0.).all(axis=1)]
print(df)
# Data Processing
df.iloc[:, 0] = np.log(df.iloc[:, 0])  # Logarithmic transformation of Max_IDR
df.iloc[:, 1] = np.log(df.iloc[:, 1])  # Logarithmic transformation of Max_PFA
df.iloc[:, 2] = np.log(df.iloc[:, 2])  # Logarithmic transformation of Top_PFA
df.iloc[:, 3] = np.log(df.iloc[:, 3])  # Logarithmic transformation of top_IDR
df.iloc[:, 4] = np.log(df.iloc[:, 4])  # Logarithmic transformation of nStory
df.iloc[:, 5] = np.log(df.iloc[:, 5])  # Logarithmic transformation of story height
df.iloc[:, 6] = np.log(df.iloc[:, 6])  # Logarithmic transformation of year
df.iloc[:, 7] = np.log(df.iloc[:, 7])  # Logarithmic transformation of stru type
df.iloc[:, 8] = np.log(df.iloc[:, 8])  # Logarithmic transformation of Earthquake Magnitude
df.iloc[:, 9] = np.log(df.iloc[:, 9])  # Logarithmic transformation of EpiD (km)
df.iloc[:, 10] = np.log(df.iloc[:, 10])  # Logarithmic transformation of Vs30
df.iloc[:, 11] = np.log(df.iloc[:, 11])  # Logarithmic transformation of PGA
df.iloc[:, 12] = np.log(df.iloc[:, 12])  # Logarithmic transformation of Significant_Duration
df.iloc[:, 13] = np.log(df.iloc[:, 13])  # Logarithmic transformation of Arias_Intensity
df.iloc[:, 14] = np.log(df.iloc[:, 14])  # Logarithmic transformation of Cumulative_Absolute_Velocity
df.iloc[:, 15] = np.log(df.iloc[:, 15])  # Logarithmic transformation of Peak_Ground_Velocity (PGV)

# Put the processed data back into the DataFrame object.
cleaned_df = df.copy()
cleaned_df = cleaned_df.replace([float('inf'), float('-inf')], np.nan).astype(float)
cleaned_df = cleaned_df.dropna()
print("清理后的数据行数:", len(cleaned_df))
print(cleaned_df.isnull().sum())

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
# Extract the input data
X = cleaned_df.iloc[:, 4:16]
# Extract the target data (Note: cleaned_df.iloc[:, 2] is Top_PFA, cleaned_df.iloc[:, 1] is Max_PFA, and cleaned_df.iloc[:, 1] is Max_IDR)
y = cleaned_df.iloc[:, 2]
print('Input:', X)
print('Output:', y)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

torch.manual_seed(1)
np.random.seed(1)

# Divide the data into training set, validation set, and test set
X_train, X_all, y_train, y_all = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Create DataLoader for training set, validation set, and test set data
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# Build the EEWnet Model
class EEWnet(nn.Module):
    def __init__(self):
        super(EEWnet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(X_train.shape[1], 512),
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
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(32),
            nn.Dropout(0),

            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Instantiate the model
model = EEWnet().to(device)
# Select the type of loss function
criterion = nn.MSELoss()
# Select the type of optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


# Training Process
def train_model(epochs):
    train_losses = []
    val_losses = []
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
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        if val_losses[epoch] < best_loss:
            best_loss = val_losses[epoch]
            save_path = os.path.join('model', 'my_best_model.pth')
            torch.save(model.state_dict(), save_path)
        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Test Loss: {val_loss / len(val_loader)}')
    # Create loss.txt to store training and validation losses
    loss_file = os.path.join('model', 'loss.txt')
    with open(loss_file, 'w') as file:
        file.write('train_loss' + '\t' + 'val_loss' + '\n')
        for i in range(len(train_losses)):
            file.write(str(train_losses[i]) + '\t' + str(val_losses[i]) + '\n')
    # Generate the image of training loss and validation loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('model', f"loss_{current_time}.png"))
    plt.show()
    return model


trained_model = train_model(epochs)


# Test Model
def test_model(model, test_loader):
    model.load_state_dict(torch.load('./model/my_best_model.pth'))
    model.eval()
    predicted = []
    actual = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predicted.extend(output.view(-1).tolist())
            actual.extend(y_batch.view(-1).tolist())

    # Calculate the MAE (Mean Absolute Error)
    mae = mean_absolute_error(actual, predicted)
    print(f'Mean Absolute Error (MAE): {mae}')

    # Calculate the MSE (Mean Squared Error)
    mse = mean_squared_error(actual, predicted)
    print(f'Mean Squared Error (MSE): {mse}')

    # Calculate the RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    # Draw a scatter plot of predicted values and actual values
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5)
    # 45 degree line
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r')
    # 'Target output' Please fill in the EDP parameter (MIDR or MPFA or PTFA) output by the model
    plt.xlabel('Actual In(Target output) In(unit)')
    plt.ylabel('Predicted In(Target output) In(unit)')
    plt.title(f'R^2 Score: {r2_score(actual, predicted):.4f}')
    plt.savefig(f"Predicted_vs_actual_{current_time}.png")
    plt.show()

    # Create files for actual values and predicted values to save the actual values and predicted values
    actual_predicted_file = os.path.join('model', 'Actual_Predicted.txt')
    with open(actual_predicted_file, 'w') as file:
        file.write('Actual' + '\t' + 'Predicted' + '\n')
        for i in range(len(actual)):
            file.write(str(actual[i]) + '\t' + str(predicted[i]) + '\n')
    # Create a file to save the metrics
    metrics_file = os.path.join('model', 'metrics.txt')
    with open(metrics_file, 'w') as file:
        file.write('MAE =' + str(mae) + '\n' + 'MSE =' + str(mse) + '\n' + 'RMSE =' + str(rmse))


test_model(trained_model, test_loader)