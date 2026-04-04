from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict

class BaseLoader(ABC):
    
    @abstractmethod
    def load(self) -> None:
        ...
    
    @abstractmethod
    def get_episode(self, task_name: str, episode_length: int) -> pd.DataFrame:
        ...
    
    @abstractmethod
    def get_label(self, record: pd.Series) -> str:
        ...
    
    @abstractmethod
    def label_counts(self) -> Dict[str, int]:
        ...

