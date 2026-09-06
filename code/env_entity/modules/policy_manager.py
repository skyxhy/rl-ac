import bisect
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

class PolicyManager:
    def __init__(self,df:pd.DataFrame,init_access_level = 1):
        self.init_access_level = init_access_level

    def reset(self):
        self.level_map:Dict[int,int] = {}

    def set(self, pseudo:int, level:int):
        self.level_map[pseudo] = level

    def get(self, pseudo:int):
        return self.level_map.get(pseudo, self.init_access_level)
    
    def batch_set(self, pseudo_level_map:Dict[int,int]):
        for pseudo, level in pseudo_level_map.items():
            self.set(pseudo, level)
    def batch_set(self, pseudo_map:Dict[int,int],levels:np.ndarray):
        for pseudo, i in pseudo_map.items():
            self.set(pseudo, levels[i])
    def batch_get(self, pseudos:List[int])->List[int]:
        return [self.get(pseudo) for pseudo in pseudos]
