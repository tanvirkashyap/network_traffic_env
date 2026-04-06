from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from data.base_loader import BaseLoader

TASK_ATTACKS = {
    "obvious": ["normal.", "nmap.", "neptune.", "back.", "portsweep."],
    "subtle": ["normal.", "ipsweep.", "satan.", "buffer_overflow.", "nmap.", "neptune.", "back.", "portsweep."],
    "mixed": None
}

class KDDLoader(BaseLoader):
    
    def __init__(self):
        self.df = None
        self.load()
    
    def load(self) -> None:
        data = fetch_kddcup99()
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        self.df['label'] = data.target
    
    # decode all columns from bytes to strings
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].str.decode('utf-8')
    
    # convert numeric columns to numbers
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except:
                pass
    
    # label encode categorical columns
        le = LabelEncoder()
        for col in ['protocol_type', 'service', 'flag']:
            self.df[col] = le.fit_transform(self.df[col])
    
    # add is_attack column
        self.df['is_attack'] = self.df['label'].apply(
            lambda x: 0 if x == 'normal.' else 1
    )
        # keep only rows where label is one of these values
        
       # Drop duplicates
        self.df = self.df.drop_duplicates()

# Handle missing values (just in case)
        self.df = self.df.dropna()

# Reset index
        self.df = self.df.reset_index(drop=True)
    
    def get_episode(self, task_name: str, episode_length: int) -> pd.DataFrame:
        allowed_labels = TASK_ATTACKS[task_name]
    
        if allowed_labels is None:
            filtered_df = self.df
        else:
            filtered_df = self.df[self.df['label'].isin(allowed_labels)]
    
        episode_length = min(episode_length, len(filtered_df))
        return filtered_df.sample(n=episode_length)
    
    def get_label(self, record: pd.Series) -> str:
        return "normal" if record['label'] == "normal." else "attack"

    def label_counts(self) -> Dict[str, int]:
        return self.df["label"].value_counts().to_dict()