CONFIG = {
    "task_name": "subtle",
    "episode_length": 100,
    "episodes": 10
}

def grader(history):
    if not history:
        return 0.0
    total_reward = sum(step["reward"] for step in history)
    max_possible = len(history) * 1.0
    # subtract penalty for missed attacks (stricter than task 1)
    missed = sum(1 for step in history if step["reward"] == -1.0)
    penalty = missed * 0.1
    score = (total_reward / max_possible) - penalty
    return max(0.0, score)