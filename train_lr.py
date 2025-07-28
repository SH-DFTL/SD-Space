import torch
import torch_npu
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.MyDataset import MyDataset
from models.CNN_regressor import CNNRegressor
from models.EnhancedRegress import EnhancedVectorRegressor
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch_npu.contrib import transfer_to_npu
#from torch_npu.npu import amp
from tqdm import tqdm
from torch.optim import lr_scheduler

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

# 配置参数
class Config:
    # 数据路径
    X_PATH = "/home/data/dataset/clean/X_cleaned.npy"
    Y_PATH = "/home/data/dataset/clean/y_cleaned.npy"
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 300
    LEARNING_RATE = 0.001
    SPLIT_RATIOS = [0.7, 0.15, 0.15]  # 训练/验证/测试
    
    # 模型保存
    SAVE_DIR = "runs/exp_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    MODEL_NAME = "model.pth"
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 创建保存目录
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # 加载数据集
    print("Loading the Dataset ...")
    X = np.load(Config.X_PATH)
    y = np.load(Config.Y_PATH)
    # dataset = MyDataset(features, labels)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 3. 创建数据集对象
    train_dataset = MyDataset(X_train, y_train)  # 训练集创建归一化器
    val_dataset = MyDataset(X_val, y_val)  # 使用训练集的归一化器
    test_dataset = MyDataset(X_test, y_test)  # 使用训练集的归一化器
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    print("Start to load model")
    # 初始化模型
    #model = CNNRegressor(input_dim=5).to(Config.DEVICE)
    model = EnhancedVectorRegressor().to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',          # 监控验证损失，越小越好
        factor=0.5,         # 学习率衰减因子
        patience=5,         # 等待多少个epoch没有改善
        verbose=True,       # 打印学习率变化
        threshold=0.0001,   # 最小变化量才认为是改善
        min_lr=1e-6         # 最小学习率限制
    )   

    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    print("Start to train ... ")
    # 训练循环
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        #count = 0 
        # 训练阶段
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS} [Train]'):
            #features = batch['features'].to(Config.DEVICE)
            #labels = batch['label'].to(Config.DEVICE)
            features = features.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            #print("========================")
            #print("features",features)
            #print("labels",labels)
            #optimizer.zero_grad()
            outputs = model(features)
            #print("outputs",outputs)
            loss = criterion(outputs, labels)
            #print("loss",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * features.size(0)
            #count+=1
            #if count == 10:
            #    break        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS} [Val]'):
            #for features,labels in val_loader:
                #features = batch['features'].to(Config.DEVICE)
                #labels = batch['label'].to(Config.DEVICE)
                
                features = features.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                #print(features)
                #print(labels)

                outputs = model(features)
                loss = criterion(outputs, labels)
                #print(loss)
                epoch_val_loss += loss.item() * features.size(0)
                absolute_errors = torch.abs(outputs - labels)
                val_mae += absolute_errors.sum().item()
                #count+=1
                #if count == 20:
                #    break
                
        
        # 计算平均损失
        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            # 基于验证损失更新学习率
            scheduler.step(epoch_val_loss)
        else:
            # 其他调度器按epoch更新
            scheduler.step()
    
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
    
        print(f'Epoch {epoch+1}/{Config.EPOCHS}: '
            f'Train Loss: {epoch_train_loss:.4f} | '
            f'Val Loss: {epoch_val_loss:.4f} | '
            f'Val MAE: {val_mae:.4f} | '
            f'LR: {current_lr:.6f}')


        # 打印进度
        #print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
        #      f"Train Loss: {epoch_train_loss:.4f} | "
        #      f"Val Loss: {epoch_val_loss:.4f}")
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, Config.MODEL_NAME))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.SAVE_DIR, 'loss_curve.png'))
    plt.show()

    # ▒~]▒~X训▒~C记▒~U
    np.save(os.path.join(Config.SAVE_DIR, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(Config.SAVE_DIR, 'val_losses.npy'), np.array(val_losses))

    # 测试阶段
    test_loss = 0.0
    
    #train_losses = np.load(os.path.join(Config.SAVE_DIR, 'train_losses.npy'))
    #val_losses = np.load(os.path.join(Config.SAVE_DIR, 'val_losses.npy'))
    total_mae = 0 
    model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)))
    model.eval()
    with torch.no_grad():
        for features,labels in test_loader:
            #features = batch['features'].to(Config.DEVICE)
            #labels = batch['label'].to(Config.DEVICE)
            features = features.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * features.size(0)
            absolute_errors = torch.abs(outputs - labels)
            total_mae += absolute_errors.sum().item()
  
    test_loss /= len(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    total_mae /= len(test_loader)
    print(f"\nTest MAE: {total_mae:.4f}")
    # 绘制损失曲线
    #plt.figure(figsize=(10, 5))
    #plt.plot(train_losses, label='Train Loss')
    #plt.plot(val_losses, label='Validation Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Training and Validation Loss')
    #plt.legend()
    #plt.grid(True)
    #plt.savefig(os.path.join(Config.SAVE_DIR, 'loss_curve.png'))
    #plt.show()
    
    # 保存训练记录
    #np.save(os.path.join(Config.SAVE_DIR, 'train_losses.npy'), np.array(train_losses))
    #np.save(os.path.join(Config.SAVE_DIR, 'val_losses.npy'), np.array(val_losses))

if __name__ == "__main__":
    train()
