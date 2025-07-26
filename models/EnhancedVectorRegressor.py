import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedVectorRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, dropout_rate=0.15):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            dropout_rate: Dropout概率
        """
        super().__init__()

        # 输入层：添加批量归一化和激活函数
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()  # 比ReLU更平滑，有助于梯度流动
        )

        # 残差块1：跳跃连接防止梯度消失
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # 残差块2：第二个残差块
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # 自适应特征压缩
        self.adaptive_compression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate // 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU()
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim // 4, 1)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 输入处理
        x = self.input_block(x)

        # 残差块1
        residual = x
        x = self.res_block1(x)
        x = F.gelu(x + residual)  # 残差连接

        # 残差块2
        residual = x
        x = self.res_block2(x)
        x = F.gelu(x + residual)  # 残差连接

        # 特征压缩
        x = self.adaptive_compression(x)

        # 输出
        return self.output_layer(x)

    def _initialize_weights(self):
        """自定义权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier初始化，适合线性层
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
