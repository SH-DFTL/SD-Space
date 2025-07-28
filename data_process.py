import os
import glob
import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from tqdm import tqdm

def process_single_nc(file_path):
    """处理单个NC文件并返回归一化数据"""
    try:
        # 读取NC文件
        ds = xr.open_dataset(file_path)
        
        # 提取特征和标签
        time_values = np.arange(24)  # 每天24小时数据
        lat_value = ds.lat.values
        lon_value = ds.lon.values
        height_values = ds.height.values
        
        # 创建特征矩阵 (n_samples, n_features)
        n_samples = len(time_values) * len(height_values)
        X = np.zeros((n_samples, 5))  # 7个特征
        
        # 填充特征矩阵
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
        
        # 提取标签
        y = ds.ELECDEN.values.reshape(-1, 1)
        
        return X, y
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None

def process_and_save_all(input_dir, output_dir, num_processes=4):
    """处理所有NC文件并保存为单个NPY文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有NC文件路径
    nc_files = glob.glob(os.path.join(input_dir, "**/*.nc"), recursive=True)#[:1]
    
    # 使用多进程处理所有文件
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_single_nc, nc_files), total=len(nc_files)))
    
    # 合并所有有效结果
    all_X = []
    all_y = []
    for X, y in results:
        if X is not None and y is not None:
            all_X.append(X)
            all_y.append(y)
    
    # 合并为单个数组
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    # 归一化处理
    X_scaler = MinMaxScaler()
    X_normalized = X_scaler.fit_transform(X_combined).astype(np.float16)
    
    y_scaler = MinMaxScaler()
    y_normalized = y_scaler.fit_transform(y_combined).astype(np.float16)
    
    # 保存合并后的数据和归一化参数
    np.save(os.path.join(output_dir, "combined_X.npy"), X_normalized)
    np.save(os.path.join(output_dir, "combined_y.npy"), y_normalized)
    np.save(os.path.join(output_dir, "X_scaler_params.npy"), 
            np.array([X_scaler.min_, X_scaler.scale_], dtype=object))
    np.save(os.path.join(output_dir, "y_scaler_params.npy"), 
            np.array([y_scaler.min_, y_scaler.scale_], dtype=object))
    
    print(f"处理完成！共处理{len(nc_files)}个NC文件，最终得到{X_normalized.shape[0]}个样本")

if __name__ == "__main__":
    # 使用示例
    input_directory = "/home/data/WACCM.ELECDEN.BEIJING.2002-2023"  # 替换为你的NC文件目录
    output_directory = "./output"  # 替换为输出目录
    
    process_and_save_all(input_directory, output_directory, num_processes=8)
