import requests
from typing import Optional

from models import NetworkAction, NetworkObservation



class NetworkEnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, task_name="obvious"):
        response = requests.post(
        f"{self.base_url}/reset",
        json={"task_name": task_name}   
    )
        response.raise_for_status()
        return NetworkObservation(**response.json())
    # -------------------------
    # STEP
    # -------------------------
    def step(self, action: NetworkAction) -> NetworkObservation:
        response = requests.post(
        f"{self.base_url}/step",
        json=self._step_payload(action)
    )
        response.raise_for_status()
        return self._parse_observation(response.json())

    # -------------------------
    # STATE
    # -------------------------
    def state(self):
        response = requests.get(f"{self.base_url}/state")
        return response.json()

    # -------------------------
    # INTERNAL: ACTION → JSON
    # -------------------------
    def _step_payload(self, action: NetworkAction):
        return {
        "action_id": int(action.action_id)
    }

    # -------------------------
    # INTERNAL: JSON → OBSERVATION
    # -------------------------
    def _parse_observation(self, data) -> NetworkObservation:
        return NetworkObservation(
            duration=data["duration"],
            protocol_type=data["protocol_type"],
            service=data["service"],
            flag=data["flag"],
            src_bytes=data["src_bytes"],
            dst_bytes=data["dst_bytes"],
            wrong_fragment=data["wrong_fragment"],
            urgent=data["urgent"],
            num_failed_logins=data["num_failed_logins"],
            hot=data["hot"],

            window_avg_src_bytes=data["window_avg_src_bytes"],
            window_avg_duration=data["window_avg_duration"],
            window_same_src_count=data["window_same_src_count"],
            window_attack_count=data["window_attack_count"],

            done=data["done"],
            reward=data.get("reward"),
            step=data["step"]
        )