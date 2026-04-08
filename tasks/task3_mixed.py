CONFIG = {
    "task_name": "mixed",
    "episodes": 10
}

def grader(history):
    penalty = 0
    reward = 0

    for step in history:
        if step["is_attack"] and step["action"] == 1:
            reward += 1
        elif not step["is_attack"] and step["action"] == 1:
            penalty += 1  # false positives hurt more

    score = reward - penalty
    return max(0.0, score / len(history))# Mixed attacks - spy, rootkit, warezmaster

