# Network Traffic Security Environment (OpenEnv)

## Overview
This project implements a **real-world reinforcement learning environment** for network intrusion detection and response.  

An agent observes network traffic features and decides whether to:
- allow traffic  
- block traffic  
- ignore traffic  

The goal is to **maximize security (catch attacks)** while **minimizing false positives**.

---

## Motivation
Modern cybersecurity systems must:
- Detect malicious activity in real time  
- Avoid blocking legitimate users  
- Adapt to evolving attack patterns  

This environment simulates that decision-making process, making it suitable for:
- Reinforcement Learning research  
- Security-focused AI agents  
- Real-world decision systems  

---

## Environment Design

### Observation Space
Each step provides structured features from network traffic:

- `duration`
- `protocol_type`
- `service`
- `flag`
- `src_bytes`, `dst_bytes`
- `wrong_fragment`, `urgent`
- `num_failed_logins`, `hot`

#### Window-based features:
- `window_avg_src_bytes`
- `window_avg_duration`
- `window_same_src_count`
- `window_attack_count`

#### Additional:
- `step`
- `reward`
- `done`

---

### Action Space

| Action ID | Description |
|----------|------------|
| 0 | Allow traffic |
| 1 | Flag for review |
| 2 | Block traffic |

---

### Reward Function

| Scenario | Reward |
|--------|--------|
| Correctly block attack          | +1.0 |
| Flag an attack                  | +0.5 |
| Correctly allow normal          | +0.3 |
| Unnecessary flag                | -0.1 |
| False positive (block normal)   | -0.5 |
| Missed attack (allow attack)    | -1.0 |

This provides **dense feedback** across the episode.

---

## Tasks

The environment includes 3 tasks of increasing difficulty:

| Task | Difficulty | Description |
|------|----------|------------|
| `obvious` | Easy   | Clear patterns — nmap, neptune, portsweep        |
| `subtle`  | Medium | Mixed signals — ipsweep, satan, buffer_overflow  |
| `mixed`   | Hard   | All attack types including rare ones             |

Each task includes a deterministic **grader** returning a score between `0.0 – 1.0`.

---

## API Endpoints

The environment is exposed via a FastAPI server:
POST /reset → start new episode
POST /step → take action
GET /state → environment state


## Baseline

Run:

python baseline.py

Example output:

Score (obvious): 0.52
Score (subtle):  0.41
Score (mixed):   0.28
## Docker
docker build -t network-env .
docker run -p 7860:7860 network-env

##Deployment

Deployed on Hugging Face Spaces (Docker-based).

## Setup

### Local Development
```bash
git clone https://github.com/yourusername/network_traffic_env
cd network_traffic_env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Using the Client
```python
from client import NetworkEnvClient
from models import NetworkAction

env = NetworkEnvClient(base_url="http://localhost:7860")
obs = env.reset(task_name="obvious")

while not obs.done:
    action = NetworkAction(action_id=2)  # always block
    obs = env.step(action)

print("Episode done")
```

### Running Inference with LLM
```bash
export OPENAI_API_KEY=your_key_here
python inference.py
```
