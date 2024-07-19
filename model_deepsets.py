import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSetsAttClass(nn.Module):
    def __init__(self, num_feat, num_heads=4, num_transformer=4, projection_dim=32):
        super(DeepSetsAttClass, self).__init__()
        self.num_feat = num_feat
        self.num_heads = num_heads
        self.num_transformer = num_transformer
        self.projection_dim = projection_dim
        
        self.time_distributed1 = nn.Linear(num_feat, projection_dim)
        self.time_distributed2 = nn.Linear(projection_dim, projection_dim)
        self.attention_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=projection_dim, 
                nhead=num_heads, 
                dropout=0.1
            ) for _ in range(num_transformer)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dense1 = nn.Linear(projection_dim, 2 * projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dense2 = nn.Linear(2 * projection_dim, 1)
        
    def forward(self, x):
        x = self.time_distributed1(x)
        x = F.leaky_relu(x, 0.01)
        x = self.time_distributed2(x)
        
        for layer in self.attention_layers:
            x = layer(x)
        
        x = x.permute(0, 2, 1)  # Change shape to [batch, projection_dim, num_const]
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.leaky_relu(x)
        x = self.dense2(x)
        
        return torch.sigmoid(x)
