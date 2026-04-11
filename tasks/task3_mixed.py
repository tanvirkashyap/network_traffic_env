CONFIG = {
    "task_name": "mixed",
    "episode_length": 100,
    "episodes": 10
}

def grader(history):
    if not history: return 0.0
    total_reward = sum(step["reward"] for step in history)
    # 0.0 = Missed Attack, 0.1 = False Positive (Blocking normal)
    missed = sum(1 for step in history if step["reward"] == 0.0)
    false_pos = sum(1 for step in history if step["reward"] == 0.1)
    penalty = (missed * 0.1) + (false_pos * 0.05)
    score = (total_reward / len(history)) - penalty
    return round(max(0.0, min(1.0, score)), 3)