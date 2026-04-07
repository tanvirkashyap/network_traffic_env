from turtle import done, pd

from data.kdd_loader import KDDLoader
from models import NetworkAction, NetworkObservation, NetworkState

class NetworkEnvironment:
    def __init__(self):
        self.loader = KDDLoader()
        self.data = None
        self.current_step = 0

    def reset(self):
        self.episode_data = self.loader.get_episode(self.task_name, self.episode_length)
        self.current_step = 0
        self.window = []
        self.episode_id = f"{self.task_name}_{np.random.randint(1e6)}"

    # reset metrics
        self.correct_blocks = 0
        self.false_positives = 0
        self.missed_attacks = 0

        first_row = self.episode_data.iloc[self.current_step]
        return self._build_observation(first_row, reward=None, done=False)
    

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
    # map row → NetworkObservation
    
    def step(self, action):
        row = self.episode_data.iloc[self.current_step]
        is_attack = row['is_attack']

        reward = self._calculate_reward(action.action_id, is_attack)

    # update metrics
        if action.action_id == 1:  # block
            if is_attack:
                self.correct_blocks += 1
            else:
                self.false_positives += 1
        elif action.action_id == 0:  # allow
            if is_attack:
                self.missed_attacks += 1

    # update sliding window
        self.window.append(row)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    # move forward
        self.current_step += 1
        done = self.current_step >= len(self.episode_data)

        if not done:
            next_row = self.episode_data.iloc[self.current_step]
        else:
            next_row = row  # dummy reuse for final observation

        obs = self._build_observation(next_row, reward=reward, done=done)

        return obs
    
    def _calculate_reward(self, action_id, is_attack):
        if action_id == 1:  # block
            return 1.0 if is_attack else -0.5

        elif action_id == 0:  # allow
            return -1.0 if is_attack else 0.3

        elif action_id == 2:  # flag
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


