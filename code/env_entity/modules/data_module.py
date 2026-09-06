import pandas as pd
import numpy as np
from typing import Dict

class DataModule:
    def __init__(
        self,
        data_path='output.csv'
    ):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.min_time, self.max_time = self.df.time.min(), self.df.time.max()
    
    def read_data(self,attack_intensity:float|int = 1,sender_intensity_map:Dict=None,start_time:int=None,end_time:int=None) -> pd.DataFrame:
        if start_time is None or end_time is None:
            start_time = self.min_time
            end_time = self.max_time
        if start_time < self.min_time or end_time > self.max_time:
            raise ValueError("Invalid time range")
        df = load_window_with_attack_intensity(
            self.df,
            start_time,
            end_time,
            attack_intensity,
            sender_intensity_map=sender_intensity_map
        )

        return df

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

def load_window_with_attack_intensity(
    df: pd.DataFrame,
    start_time: int,
    end_time: int,
    attack_intensity: float = 1.0,
    attack_col: str = "attack_type",
    sender_col: str = "sender_id",
    time_col: str = "time",
    random_seed: int = 42,
    sender_intensity_map: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """
    根据攻击强度读取指定时间窗口数据，并对攻击节点请求进行比例采样。
    支持传入节点级攻击强度字典，未配置的节点默认使用全局强度。

    Parameters
    ----------
    df : pd.DataFrame
        原始完整数据
    start_time : int
        时间窗口起始（包含）
    end_time : int
        时间窗口结束（不包含）
    attack_intensity : float
        全局攻击强度，表示攻击节点请求保留比例 (0, 1]
    attack_col : str
        攻击标签列名
    sender_col : str
        发送方ID列名
    time_col : str
        时间列名
    random_seed : int
        随机种子
    sender_intensity_map : Optional[Dict[Any, float]]
        节点级攻击强度映射，格式 {sender_id: intensity}。
        若为 None 或字典中不包含某 sender_id，则回退使用全局 attack_intensity。

    Returns
    -------
    pd.DataFrame
        处理后的时间窗口数据
    """
    # =========================
    # 参数校验
    # =========================
    assert 0 < attack_intensity <= 1.0, "全局 attack_intensity 必须在 (0, 1] 范围内"
    if sender_intensity_map is not None:
        for sid, intensity in sender_intensity_map.items():
            assert 0 < intensity <= 1.0, f"节点 {sid} 的攻击强度必须在 (0, 1] 范围内"

    rng = np.random.default_rng(random_seed)

    # =========================
    # 1. 时间窗口截取
    # =========================
    window_df = df[
        (df[time_col] >= start_time) &
        (df[time_col] < end_time)
    ].copy()

    if window_df.empty:
        return window_df

    # =========================
    # 2. 区分攻击节点和正常节点
    # =========================
    attack_mask = window_df[attack_col] > 0
    benign_df = window_df[~attack_mask]
    attack_df = window_df[attack_mask]

    if attack_df.empty:
        return window_df.reset_index(drop=True)

    # =========================
    # 3. 按 sender 粒度进行比例保留
    # =========================
    sampled_attack_parts = []

    for sender_id, group in attack_df.groupby(sender_col):
        # 优先使用节点专属强度，否则回退到全局强度
        current_intensity = sender_intensity_map.get(sender_id, attack_intensity) if sender_intensity_map else attack_intensity
        
        n_total = len(group)
        n_keep = max(1, int(np.ceil(n_total * current_intensity)))
        
        # 转换为 numpy 数组避免 pandas Index 在部分版本中的兼容性问题
        pool_idx = group.index.to_numpy()
        keep_idx = rng.choice(pool_idx, size=n_keep, replace=False)
        
        sampled_attack_parts.append(group.loc[keep_idx])

    # =========================
    # 4. 合并返回
    # =========================
    if sampled_attack_parts:
        sampled_attack_df = pd.concat(sampled_attack_parts, axis=0)
    else:
        # 防御性编程：理论上不会走到这里（attack_df 非空时必有分组）
        sampled_attack_df = pd.DataFrame(columns=attack_df.columns)

    out_df = pd.concat([benign_df, sampled_attack_df], axis=0)
    out_df = out_df.sort_values(time_col).reset_index(drop=True)

    return out_df

