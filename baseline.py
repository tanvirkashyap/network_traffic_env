import numpy as np
from client import NetworkEnvClient
from models import NetworkAction
import os

# import graders
from tasks.task1_obvious import grader as grader1
from tasks.task2_subtle import grader as grader2
from tasks.task3_mixed import grader as grader3


def run_episode(task_name):
    
    hf_url = os.getenv("API_BASE_URL", "https://tanananana-network-traffic-env.hf.space")
    
    client = NetworkEnvClient(base_url=hf_url)
    obs = client.reset(task_name=task_name)

    history = []

    max_steps = 1000
    steps = 0

    while not obs.done and steps < max_steps:
        print("Step:", obs.step, "Done:", obs.done)

        action_id = np.random.choice([0, 1, 2])
        action = NetworkAction(action_id=action_id)

        history.append({
            "action": action_id,
            "reward": obs.reward if obs.reward is not None else 0.0
        })

        obs = client.step(action)
        steps += 1
    #print("Step:", steps, "Done:", obs.done) 
    #print("Finished episode")

    return history


def evaluate_task(task_name, grader):
    scores = []

    for _ in range(1):  # multiple episodes for stability
        history = run_episode(task_name)
        score = grader(history)
        scores.append(score)

    return np.mean(scores)


if __name__ == "__main__":
    np.random.seed(42)

    print("Starting Baseline Evaluation...")

    # Task 1
    print("\nEvaluating Task 1 (Obvious)...")
    score1 = evaluate_task("obvious", grader1)
    print(f"Task 1 Score: {score1:.3f}")

    # Task 2
    print("\nEvaluating Task 2 (Subtle)...")
    score2 = evaluate_task("subtle", grader2)
    print(f"Task 2 Score: {score2:.3f}")

    # Task 3
    print("\nEvaluating Task 3 (Mixed)...")
    score3 = evaluate_task("mixed", grader3)
    print(f"Task 3 Score: {score3:.3f}")

    print("\n=========================")
    print("ALL TASKS COMPLETE")
    print(f"Final Scores: T1: {score1:.3f}, T2: {score2:.3f}, T3: {score3:.3f}")
    print("=========================")