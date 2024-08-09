# inference.py
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import Autoencoder

def load_model(model_path, input_dim):
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def detect_anomaly(data, model, scaler, threshold):
    data_normalized = scaler.transform(data)
    data_tensor = torch.FloatTensor(data_normalized)
    
    with torch.no_grad():
        reconstructed = model(data_tensor)
    
    mse = torch.mean((data_tensor - reconstructed) ** 2, axis=1)
    return mse.numpy() > threshold

# モデルのロード
input_dim = 20  # トレーニング時と同じ次元数
model = load_model('autoencoder.pth', input_dim)

# スケーラーの準備（実際のアプリケーションでは、トレーニング時のスケーラーを使用します）
scaler = StandardScaler()
scaler.fit(np.random.randn(1000, input_dim))  # ダミーデータでフィット

# テストデータの生成（実際のアプリケーションでは、実データを使用します）
normal_test = np.random.randn(100, input_dim)
anomaly_test = np.random.randn(100, input_dim) * 2 + 5  # 異常データ

# 異常検知の実行
threshold = 0.1  # この閾値は実験的に決定する必要があります
normal_results = detect_anomaly(normal_test, model, scaler, threshold)
anomaly_results = detect_anomaly(anomaly_test, model, scaler, threshold)

print(f"正常データで異常と判定された割合: {np.mean(normal_results):.2f}")
print(f"異常データで異常と判定された割合: {np.mean(anomaly_results):.2f}")
