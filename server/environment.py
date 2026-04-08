
from typing import Optional
import numpy as np
import pandas as pd

from data.kdd_loader import KDDLoader
from models import NetworkAction, NetworkObservation, NetworkState


class NetworkEnvironment:
    def __init__(self, task_name: str = "obvious", episode_length: int = 100, window_size: int = 10):
        self.loader = KDDLoader()
        self.task_name = task_name
        self.episode_length = episode_length
        self.window_size = window_size

        self.episode_data: pd.DataFrame = pd.DataFrame()
        self.current_step = 0
        self.window = []
        self.episode_id = None

        self.correct_blocks = 0
        self.false_positives = 0
        self.missed_attacks = 0

    def reset(self, task_name=None):
        if task_name:
            self.task_name = task_name
        self.episode_data = self.loader.get_episode(self.task_name, self.episode_length)
        self.current_step = 0
        self.window = []
        self.episode_id = f"{self.task_name}_{np.random.randint(1e6)}"

        self.correct_blocks = 0
        self.false_positives = 0
        self.missed_attacks = 0
        

        first_row = self.episode_data.iloc[self.current_step]
        return self._build_observation(first_row, reward=None, done=False)

    def step(self, action: NetworkAction) -> NetworkObservation:
        row = self.episode_data.iloc[self.current_step]
        is_attack = int(row["is_attack"])
    
        reward = self._calculate_reward(action.action_id, is_attack)
    
    # metrics tracking
        if action.action_id == 2:      # block
            if is_attack:
                self.correct_blocks += 1
            else:
                self.false_positives += 1
        elif action.action_id == 0:    # allow
            if is_attack:
                self.missed_attacks += 1
    
    # update window
        self.window.append(row)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
        self.current_step += 1
        done = self.current_step >= len(self.episode_data)
    
        if not done:
            next_row = self.episode_data.iloc[self.current_step]
        else:
            next_row = row
    
        return self._build_observation(next_row, reward=reward, done=done)

    def _calculate_reward(self, action_id: int, is_attack: int) -> float:
        
    
        if is_attack == 1:  # MALICIOUS CONNECTION
            if action_id == 2:    # Block (Correct)
                return 1.0
            elif action_id == 1:  # Flag (Partial Progress - detected but not stopped)
                return 0.6
            else:                # Allow (Catastrophic failure)
                return 0.0
            
        else:  # NORMAL CONNECTION
            if action_id == 0:    # Allow (Correct)
                return 0.8
            elif action_id == 1:  # Flag (Small penalty - wasted analyst time)
                return 0.4
            else:                # Block (Destructive action - blocked a real user)
                return 0.1

        return 0.0  

    def _get_window_stats(self):
        if not self.window:
            return 0.0, 0.0, 0, 0

        df = pd.DataFrame(self.window)

        if len(df) == 0:
            return 0.0, 0.0, 0, 0

        current_protocol = self.window[-1]['protocol_type']
        same_src_count = len(df[df['protocol_type'] == current_protocol])



        return (
            float(df['src_bytes'].mean()),
            float(df['duration'].mean()),
            int(same_src_count),  
            int(df['is_attack'].sum())
)

    def _build_observation(self, row: pd.Series, reward: Optional[float] = None, done: bool = False) -> NetworkObservation:
        avg_src_bytes, avg_duration, same_src_count, attack_count = self._get_window_stats()

        return NetworkObservation(
            duration=int(row['duration']),
            protocol_type=int(row['protocol_type']),
            service=int(row['service']),
            flag=int(row['flag']),
            src_bytes=int(row['src_bytes']),
            dst_bytes=int(row['dst_bytes']),
            wrong_fragment=int(row['wrong_fragment']),
            urgent=int(row['urgent']),
            num_failed_logins=int(row['num_failed_logins']),
            hot=int(row['hot']),

            window_avg_src_bytes=float(avg_src_bytes),
            window_avg_duration=float(avg_duration),
            window_same_src_count=int(same_src_count),
            window_attack_count=int(attack_count),
            

        done=bool(done),
        reward=reward,
        step=self.current_step
    )

    def state(self):
        return NetworkState(
            episode_id=self.episode_id,
            step_count=self.current_step,
            total_events=len(self.episode_data),
            task_name=self.task_name,
            correct_blocks=self.correct_blocks,
            false_positives=self.false_positives,
            missed_attacks=self.missed_attacks
        )