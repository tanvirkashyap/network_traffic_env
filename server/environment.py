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

        self.episode_data = None
        self.current_step = 0
        self.window = []
        self.episode_id = None

        self.correct_blocks = 0
        self.false_positives = 0
        self.missed_attacks = 0

    def reset(self):
        self.episode_data = self.loader.get_episode(self.task_name, self.episode_length)
        self.current_step = 0
        self.window = []
        self.episode_id = f"{self.task_name}_{np.random.randint(1e6)}"

        self.correct_blocks = 0
        self.false_positives = 0
        self.missed_attacks = 0

        first_row = self.episode_data.iloc[self.current_step]
        return self._build_observation(first_row, reward=None, done=False)

    def step(self, action):
        row = self.episode_data.iloc[self.current_step]
        is_attack = row['is_attack']

        reward = self._calculate_reward(action.action_id, is_attack)

        # metrics
        if action.action_id == 1:
            if is_attack:
                self.correct_blocks += 1
            else:
                self.false_positives += 1

        elif action.action_id == 0:
            if is_attack:
                self.missed_attacks += 1

        # window
        self.window.append(row)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # next step
        self.current_step += 1
        done = self.current_step == len(self.episode_data) - 1

        if not done:
            next_row = self.episode_data.iloc[self.current_step]
        else:
            next_row = row

        obs = self._build_observation(next_row, reward=reward, done=done)
        return obs

    def _calculate_reward(self, action_id, is_attack):
        if action_id == 1:
            return 1.0 if is_attack else -0.5
        elif action_id == 0:
            return -1.0 if is_attack else 0.3
        elif action_id == 2:
            return 0.5 if is_attack else -0.1
        return 0.0

    def _get_window_stats(self):
        if not self.window:
            return 0.0, 0.0, 0, 0

        df = pd.DataFrame(self.window)

        return (
            df['src_bytes'].mean(),
            df['duration'].mean(),
            len(df),
            df['is_attack'].sum()
        )

    def _build_observation(self, row, reward=None, done=False):
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

            done=done,
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