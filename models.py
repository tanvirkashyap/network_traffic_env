# Action, Observation, State
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class NetworkAction:
    action_id : int
    metadata : Optional[Dict[str, Any]]

@dataclass
class NetworkObservation:
    # current event (from KDD row)
    duration: int
    protocol_type: int
    service: int
    flag: int
    src_bytes: int
    dst_bytes: int
    wrong_fragment: int
    urgent: int
    num_failed_logins: int
    hot: int
    
    # window summary (calculated from last 10 events)
    window_avg_src_bytes: float
    window_avg_duration: float
    window_same_src_count: int
    window_attack_count: int
    
    # episode control (the RL loop needs these)
    done: bool
    reward: Optional[float]
    step: int
    
    # legal actions (always [0,1,2] but needs special handling)
    legal_actions: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.legal_actions:
            self.legal_actions = [0, 1, 2]
@dataclass
class NetworkState:
    episode_id: str
    step_count: int
    total_events: int
    task_name: str
    correct_blocks: int
    false_positives: int
    missed_attacks: int
