CONFIG = {
    "task_name": "subtle",
    "episodes": 10
}

def grader(history):
    score = 0

    for step in history:
        if step["is_attack"] and step["action"] == 1:
            score += 1
        elif not step["is_attack"] and step["action"] == 0:
            score += 0.5  # stricter

    return score / len(history) if history else 0.0