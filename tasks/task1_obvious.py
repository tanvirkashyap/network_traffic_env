from server.app import step


CONFIG = {
    "task_name": "obvious",
    "episodes": 10
}

def grader(history):
    correct = 0
    total = len(history)

    for step in history:
        if step["is_attack"] == 1 and step["action"] == 1:
            correct += 1

    return correct / total if total > 0 else 0.0