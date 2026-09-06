
from torch import nn
import torch
import numpy as np
import os

class CBNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, num_actions=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)
    
class CBAgent:
    def __init__(
        self,
        feature_dim,
        hidden_dim=12,
        action_dim=2,
        lr=1e-3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        device="cpu"
    ):
        self.device = torch.device(device)

        self.action_dim = action_dim

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_net = CBNet(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_actions=action_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=lr
        )

        self.loss_func = nn.SmoothL1Loss()

    # ======================
    # 和 DRQN 完全一致接口
    # ======================
    def take_action(self, state):
        if state is None:
            return {}

        node_map = state[0]
        state_tensor = torch.tensor(state[1], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        actions = []
        for pseudo, idx in node_map.items():
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = torch.argmax(q_values[idx]).item()
            actions.append(action)

        actions = np.array(actions)
        return (node_map, actions)

    # ======================
    # 关键区别在这里
    # ======================
    def update(self, bs, ba, br, bn=None):
        """
        bn 不使用，仅为了接口一致
        """
        bs = torch.tensor(bs, dtype=torch.float32, device=self.device)
        ba = torch.tensor(ba, dtype=torch.long, device=self.device)
        br = torch.tensor(br, dtype=torch.float32, device=self.device)

        q_values = self.q_net(bs)
        q_sa = q_values.gather(1, ba.unsqueeze(1)).squeeze(1)

        # ⚠ 没有 TD target
        loss = self.loss_func(q_sa, br)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def eval(self):
        self.q_net.eval()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.q_net.state_dict(),
            "epsilon": self.epsilon
        }, path)