import taichi as ti
from typing import List


@ti.data_oriented
class TiInteraction:
    def __init__(self):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
    
    def get_state(self) -> List[dict]:
        raise NotImplementedError
    
    def set_state(self, states: List[dict]) -> None:
        raise NotImplementedError


@ti.data_oriented
class TiObject:
    def __init__(self):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
    
    def get_state(self) -> List[dict]:
        raise NotImplementedError
    
    def set_state(self, states: List[dict]) -> None:
        raise NotImplementedError
