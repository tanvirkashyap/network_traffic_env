import numpy as np
from client import NetworkEnvClient
from models import NetworkAction

# import graders
from tasks.task1_obvious import grader as grader1
from tasks.task2_subtle import grader as grader2
from tasks.task3_mixed import grader as grader3


def run_episode(task_name):
    client = NetworkEnvClient()

    # reset environment
    obs = client.reset(task_name=task_name)

    history = []

    while not obs.done:
        action_id = np.random.choice([0, 1, 2])

        # store step info for grading
        history.append({
    "action": action_id,
    "reward": obs.reward,
    "done": obs.done,
    "is_attack": obs.window_attack_count > 0  # proxy until graders are built
})

        obs = client.step(NetworkAction(action_id=action_id))

    return history


def evaluate_task(task_name, grader):
    scores = []

    for _ in range(5):  # multiple episodes for stability
        history = run_episode(task_name)
        score = grader(history)
        scores.append(score)

    return np.mean(scores)


if __name__ == "__main__":
    np.random.seed(42)

    print("Running baseline...")

    score1 = evaluate_task("obvious", grader1)
    score2 = evaluate_task("subtle", grader2)
    score3 = evaluate_task("mixed", grader3)

    print("\nBaseline Scores:")
    print(f"Task 1 (Obvious): {score1:.3f}")
    print(f"Task 2 (Subtle): {score2:.3f}")
    print(f"Task 3 (Mixed):  {score3:.3f}")