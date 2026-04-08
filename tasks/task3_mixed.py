CONFIG = {
    "task_name": "mixed",
    "episode_length": 100,
    "episodes": 10
}

def grader(history):
    if not history:
        return 0.0
    total_reward = sum(step["reward"] for step in history)
    max_possible = len(history) * 1.0
    # extra penalty for both missed attacks and false positives
    missed = sum(1 for step in history if step["reward"] == -1.0)
    false_pos = sum(1 for step in history if step["reward"] == -0.5)
    penalty = (missed * 0.2) + (false_pos * 0.1)
    score = (total_reward / max_possible) - penalty
    return max(0.0, score)