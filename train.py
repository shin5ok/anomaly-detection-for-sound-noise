# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import Autoencoder

# ダミーデータの生成（実際のアプリケーションでは、実データを使用します）
def generate_dummy_data(n_samples, n_features):
    normal_data = np.random.randn(n_samples, n_features)
    return normal_data

# データの準備
n_samples, n_features = 1000, 20
data = generate_dummy_data(n_samples, n_features)

# データの正規化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# PyTorchのデータセットとデータローダーの作成
tensor_x = torch.Tensor(data_normalized)
dataset = TensorDataset(tensor_x)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデルの初期化
model = Autoencoder(n_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# トレーニングループ
num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch[0]
        
        # フォワードパス
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        # バックワードパスと最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# モデルの保存
torch.save(model.state_dict(), 'autoencoder.pth')
