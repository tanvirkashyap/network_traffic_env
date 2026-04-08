from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class NetworkAction(BaseModel):
    action_id: int
    metadata: Optional[Dict[str, Any]] = None

class NetworkObservation(BaseModel):
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
    window_avg_src_bytes: float
    window_avg_duration: float
    window_same_src_count: int
    window_attack_count: int
    done: bool
    reward: Optional[float]
    step: int
    # Use Field for defaults in Pydantic
    legal_actions: List[int] = Field(default_factory=lambda: [0, 1, 2])

class NetworkState(BaseModel):
    episode_id: str
    step_count: int
    total_events: int
    task_name: str
    correct_blocks: int
    false_positives: int
    missed_attacks: int