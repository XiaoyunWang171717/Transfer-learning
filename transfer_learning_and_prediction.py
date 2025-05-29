import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# 定义路径
BASE_DIR = 'C:/1Xiaoyun/MLP_Adducts'
PRETRAINED_MODEL_PATH = os.path.join(BASE_DIR, 'final_model.pkl')
NEW_DATA_PATH = os.path.join(BASE_DIR, 'Adduct460K_MS1_PL1A(fortransfertest).csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'transfer learning')
TRANSFERRED_MODEL_PATH = os.path.join(OUTPUT_DIR, 'transferred_model_pytorch2.pth')
DIRECT_PREDICTION_RESULT_PATH = os.path.join(OUTPUT_DIR, 'direct_prediction_results.csv')
TRANSFER_LEARNING_RESULT_PATH = os.path.join(OUTPUT_DIR, 'transfer_learning_results.csv')
PERFORMANCE_METRICS_PATH = os.path.join(OUTPUT_DIR, 'model_performance_metrics.csv')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def calculate_metrics(y_true, y_pred):
    """计算模型性能指标"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def prepare_data(data_path):
    """准备数据，将SMILES转换为分子指纹"""
    print(f"正在从 {data_path} 加载数据...")
    df = pd.read_csv(data_path)
    df = df[['SMILES', 'logNAH']]
    df = df.dropna()
    
    n_bits = 2048
    def smiles_to_fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))
    
    df['fps'] = df['SMILES'].apply(smiles_to_fingerprint)
    df = df[df['fps'].notnull()]
    X = np.stack(df['fps'].values)
    y = df['logNAH'].values.astype(float)
    
    return X, y, df

def transfer_learning(pretrained_model_path, X_train, X_test, y_train, y_test, test_df):
    """执行迁移学习"""
    print("开始迁移学习过程...")
    print(f"加载预训练模型: {pretrained_model_path}")
    
    # 加载预训练模型
    pretrained_model = joblib.load(pretrained_model_path)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # 创建新的PyTorch模型
    input_size = 2048
    hidden_sizes = pretrained_model.hidden_layer_sizes
    output_size = 1
    
    new_model = MLP(input_size, hidden_sizes, output_size)
    
    # 从预训练模型复制权重
    with torch.no_grad():
        pretrained_weights = pretrained_model.coefs_
        pretrained_biases = pretrained_model.intercepts_
        
        for i, (name, param) in enumerate(new_model.named_parameters()):
            if 'weight' in name:
                layer_idx = i // 2
                if layer_idx < len(pretrained_weights):
                    param.copy_(torch.FloatTensor(pretrained_weights[layer_idx].T))
            elif 'bias' in name:
                layer_idx = i // 2
                if layer_idx < len(pretrained_biases):
                    param.copy_(torch.FloatTensor(pretrained_biases[layer_idx]))
    
    # 冻结除最后一层外的所有层
    for param in new_model.parameters():
        param.requires_grad = False
    
    for param in new_model.network[-1].parameters():
        param.requires_grad = True
    
    # 训练模型
    criterion = nn.MSELoss()
    optimizer = optim.Adam(new_model.parameters(), lr=0.0001)
    batch_size = 32
    num_epochs = 500
    
    for epoch in range(num_epochs):
        indices = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train_tensor[batch_indices]
            batch_y = y_train_tensor[batch_indices]
            
            optimizer.zero_grad()
            outputs = new_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    new_model.eval()
    with torch.no_grad():
        y_pred = new_model(X_test_tensor)
        y_pred_np = y_pred.numpy().flatten()
        y_test_np = y_test_tensor.numpy().flatten()
        
        # 计算性能指标
        metrics = calculate_metrics(y_test_np, y_pred_np)
        print(f"迁移学习后的模型性能 - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
        
        # 保存预测结果
        result_df = test_df.copy()
        result_df['Actual_logNAH'] = y_test_np
        result_df['Predicted_logNAH'] = y_pred_np
        result_df['Error'] = y_test_np - y_pred_np
        result_df['Absolute_Error'] = np.abs(y_test_np - y_pred_np)
        
        print(f"保存迁移学习预测结果到: {TRANSFER_LEARNING_RESULT_PATH}")
        result_df.to_csv(TRANSFER_LEARNING_RESULT_PATH, index=False)
    
    # 保存模型
    print(f"保存迁移学习模型到: {TRANSFERRED_MODEL_PATH}")
    torch.save(new_model.state_dict(), TRANSFERRED_MODEL_PATH)
    return new_model, metrics

def direct_prediction(pretrained_model_path, X_test, y_test, test_df):
    """直接使用预训练模型进行预测"""
    print("开始直接预测过程...")
    print(f"加载预训练模型: {pretrained_model_path}")
    
    # 加载预训练模型
    pretrained_model = joblib.load(pretrained_model_path)
    
    # 使用预训练模型进行预测
    y_pred = pretrained_model.predict(X_test)
    
    # 计算性能指标
    metrics = calculate_metrics(y_test, y_pred)
    print(f"直接预测的性能 - R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    # 保存预测结果
    result_df = test_df.copy()
    result_df['Actual_logNAH'] = y_test
    result_df['Predicted_logNAH'] = y_pred
    result_df['Error'] = y_test - y_pred
    result_df['Absolute_Error'] = np.abs(y_test - y_pred)
    
    print(f"保存直接预测结果到: {DIRECT_PREDICTION_RESULT_PATH}")
    result_df.to_csv(DIRECT_PREDICTION_RESULT_PATH, index=False)
    
    return y_pred, metrics

if __name__ == "__main__":
    print("=== 开始执行预测和迁移学习 ===")
    print(f"预训练模型路径: {PRETRAINED_MODEL_PATH}")
    print(f"新数据路径: {NEW_DATA_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 准备数据并划分训练集和测试集
    X, y, df = prepare_data(NEW_DATA_PATH)
    
    # 使用索引进行划分，以便获取对应的原始数据
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    test_df = df.iloc[test_indices]
    
    # 执行直接预测
    print("\n=== 直接预测 ===")
    _, direct_metrics = direct_prediction(PRETRAINED_MODEL_PATH, X_test, y_test, test_df)
    
    # 执行迁移学习
    print("\n=== 迁移学习 ===")
    _, transfer_metrics = transfer_learning(PRETRAINED_MODEL_PATH, X_train, X_test, y_train, y_test, test_df)
    
    # 保存性能指标
    performance_df = pd.DataFrame({
        'Model': ['Direct Prediction', 'Transfer Learning'],
        'R2': [direct_metrics['R2'], transfer_metrics['R2']],
        'RMSE': [direct_metrics['RMSE'], transfer_metrics['RMSE']],
        'MAE': [direct_metrics['MAE'], transfer_metrics['MAE']]
    })
    
    print(f"保存模型性能指标到: {PERFORMANCE_METRICS_PATH}")
    performance_df.to_csv(PERFORMANCE_METRICS_PATH, index=False) 