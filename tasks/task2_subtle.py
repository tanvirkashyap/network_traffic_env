CONFIG = {
    "task_name": "subtle",
    "episode_length": 100,
    "episodes": 10
}

def grader(history):
    if not history: return 0.0
    total_reward = sum(step["reward"] for step in history)
    # A missed attack now gives a reward of 0.0
    missed = sum(1 for step in history if step["reward"] == 0.0)
    penalty = missed * 0.1
    score = (total_reward / len(history)) - penalty
    return max(0.0, min(1.0, score))