from urllib import response

import requests
from typing import Optional

from models import NetworkAction, NetworkObservation



class NetworkEnvClient:
    def __init__(self, base_url: str = "https://tanananana-network-traffic-env.hf.space"):
        self.base_url = base_url

    
    def reset(self, task_name="obvious"):
        response = requests.post(
        f"{self.base_url}/reset",
        json={"task_name": task_name}
    )
        response.raise_for_status()
        return self._parse_observation(response.json())
    
    def step(self, action: NetworkAction) -> NetworkObservation:
        response = requests.post(
        f"{self.base_url}/step",
        json=self._step_payload(action)
    )
        response.raise_for_status()
        return self._parse_observation(response.json())

  
    def state(self):
        response = requests.get(f"{self.base_url}/state")
        return response.json()

    def _step_payload(self, action: NetworkAction):
        return {
        "action_id": int(action.action_id)  
    }

    def _parse_observation(self, data) -> NetworkObservation:
        obs_data = data["observation"]
        
        return NetworkObservation(
            duration=obs_data["duration"],
            protocol_type=obs_data["protocol_type"],
            service=obs_data["service"],
            flag=obs_data["flag"],
            src_bytes=obs_data["src_bytes"],
            dst_bytes=obs_data["dst_bytes"],
            wrong_fragment=obs_data["wrong_fragment"],
            urgent=obs_data["urgent"],
            num_failed_logins=obs_data["num_failed_logins"],
            hot=obs_data["hot"],

            window_avg_src_bytes=obs_data["window_avg_src_bytes"],
            window_avg_duration=obs_data["window_avg_duration"],
            window_same_src_count=obs_data["window_same_src_count"],
            window_attack_count=obs_data["window_attack_count"],
            
            # These are now at the TOP level of the server response
            done=data["done"],
            reward=data.get("reward"),
            step=obs_data["step"] # Or data["info"]["step"]
        )