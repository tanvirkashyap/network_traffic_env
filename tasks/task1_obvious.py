CONFIG = {
    "task_name": "obvious",
    "episode_length": 100,
    "episodes": 10
}

def grader(history):
    if not history:
        return 0.0
    total_reward = sum(step["reward"] for step in history)
    max_possible = len(history) * 1.0
    return max(0.0, total_reward / max_possible)