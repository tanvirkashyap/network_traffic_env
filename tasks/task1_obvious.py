CONFIG = {
    "task_name": "obvious",
    "episode_length": 100,
    "episodes": 10
}

def grader(history):
    if not history: return 0.0
    # Average the rewards (which are now 0.0 to 1.0)
    score = sum(step["reward"] for step in history) / len(history)
    return round(max(0.0, min(1.0, score)), 3)