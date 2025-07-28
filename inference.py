import numpy as np
import torch
import torch_npu
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import xarray as xr
from models.CNN_regressor import CNNRegressor
from torch_npu.contrib import transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)

# 1. 加载归一化参数和模型
def load_resources(model_path, scaler_dir):
    # 加载模型
    model = CNNRegressor(input_dim=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    
    # 加载归一化参数
    X_scaler_params = np.load(f"{scaler_dir}/X_scaler_params.npy", allow_pickle=True)
    print(X_scaler_params)
    y_scaler_params = np.load(f"{scaler_dir}/y_scaler_params.npy", allow_pickle=True).item()
    
    # 重建归一化器
    X_scaler = MinMaxScaler()
    X_scaler.min_, X_scaler.scale_ = X_scaler_params['min'], X_scaler_params['scale']
    
    y_scaler = MinMaxScaler()
    y_scaler.min_, y_scaler.scale_ = y_scaler_params['min'], y_scaler_params['scale']
    
    return model, X_scaler, y_scaler

# 2. 数据预处理函数
def preprocess_input(raw_input, X_scaler):
    """
    输入: raw_input - 原始输入数据,形状为(n_samples, n_features)
    输出: 归一化后的tensor数据
    """
    # 归一化处理
    normalized_data = X_scaler.transform(raw_input)
    # 转换为tensor
    return torch.FloatTensor(normalized_data)

# 3. 推理预测函数
def predict(model, input_tensor, y_scaler):
    with torch.no_grad():
        predictions = model(input_tensor).numpy()
    # 反归一化
    return y_scaler.inverse_transform(predictions)

# 主函数
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = "/home/Workspace0722/runs/exp_20250724_034606/model.pth"
    SCALER_DIR = "/home/data/data2016npy"
    
    # 加载资源
    model, X_scaler, y_scaler = load_resources(MODEL_PATH, SCALER_DIR)
    
    # 示例输入数据 (替换为实际输入)
    
    # example_input = np.array([
    #     [1.0, 2.0, 3.0, 4.0],  # 样本1
    #     [5.0, 6.0, 7.0, 8.0]   # 样本2
    # ])
    file_path = r"/home/data/WACCM.ELECDEN.BEIJING.2002-2023/Ne2002_beijing/WACCM.ELECDEN.BEIJING.2002-01-01-00000.nc"
    ds = xr.open_dataset(file_path)

    # ▒~V~R~O~P▒~V~R~O~V▒~V~R~I▒~V~R▒~V~R~A▒~V~R~R~L▒~V~R| ~G签
    time_values = np.arange(24)  # ▒~V~R~O天24▒~V~R~O▒~V~R~W▒~V~R▒~V~R~U▒~V~R▒~V~R~M▒~V~R
    lat_value = ds.lat.values
    lon_value = ds.lon.values
    height_values = ds.height.values

    # ▒~V~R~H~[建▒~V~R~I▒~V~R▒~V~R~A▒~V~R~_▒~V~R▒~V~R~X▒~V~R (n_samples, n_features)
    n_samples = len(time_values) * len(height_values)
    X = np.zeros((n_samples, 5))  # 7个▒~V~R~I▒~V~R▒~V~R~A

    # 填▒~V~R~E~E▒~V~R~I▒~V~R▒~V~R~A▒~V~R~_▒~V~R▒~V~R~X▒~V~R
    idx = 0
    for t in time_values:
        for h in height_values:
            X[idx, 0] = t
            #X[idx, 1] = lat_value
            #X[idx, 2] = lon_value
            X[idx, 1] = h
            X[idx, 2] = ds.ap.isel(time=t).values
            X[idx, 3] = ds.f107.isel(time=t).values
            X[idx, 4] = ds.f107a.isel(time=t).values
            idx += 1

        # ▒~V~R~O~P▒~V~R~O~V▒~V~R| ~G签
        y = ds.ELECDEN.values.reshape(-1, 1)

    for input,label in X,y:
        # 预处理
        input_tensor = preprocess_input(input, X_scaler)
        
        # 预测
        raw_predictions = predict(model, input_tensor, y_scaler)
        
        print("原始预测值:", raw_predictions)
