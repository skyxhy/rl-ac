from torch import nn

# =========================================================
# MLP 模型定义
# =========================================================
class MLP_model(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def embed(self, x):
        return self.network[:-1](x)