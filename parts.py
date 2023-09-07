from typing import List

import numpy as np
class joints:
    length: int
    rot: np.ndarray
    vol: np.ndarray
    acc:np.ndarray
    children: List["joints"]
    def step(self,delta:float):
        self.rot=self.rot+self.vol*delta
        self.vol = self.vol+self.acc*delta
        for i in self.children:
            i.step(delta)


