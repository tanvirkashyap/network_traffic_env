import os
import json

from tasks.task1_obvious import grader as grader1
from tasks.task2_subtle import grader as grader2
from tasks.task3_mixed import grader as grader3

import os
from openai import OpenAI
from client import NetworkEnvClient
from models import NetworkAction

# 1. MANDATORY: The LLM Brain (pointing to Hugging Face Cloud)
LLM_API_URL = "https://api-inference.huggingface.co/v1/"
HF_TOKEN = os.getenv("HF_TOKEN")

# Use the ID from your sample code
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 2. Your Cybersecurity Environment
ENV_URL = os.getenv("ENV_URL", "https://tanananana-network-traffic-env.hf.space")

# Initialize OpenAI Client (This satisfies the hackathon requirement)
openai_client = OpenAI(
    base_url=LLM_API_URL,
    api_key=HF_TOKEN
)

# Initialize Environment
env = NetworkEnvClient(base_url=ENV_URL)

def choose_action(obs):
    # Simplified prompt for Llama 3.1
    prompt = f"Categorize this network traffic. Reply ONLY with 0 (Allow), 1 (Flag), or 2 (Block). Stats: duration={obs['duration']}, bytes={obs['src_bytes']}. Action:"

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        # Safe extraction: find the first digit
        for char in content:
            if char in ['0', '1', '2']:
                return int(char)
        return 1
    except Exception as e:
        print(f"LLM Error: {e}")
        return 1# fallback

def run_episode(task_name="obvious", max_steps=100):
    obs = env.reset(task_name=task_name)
    history = []
    
    # MANDATORY FORMAT
    print(f"[START] {task_name}")

    steps = 0
    while not obs.done and steps < max_steps:
        action_id = choose_action(obs.model_dump())
        action = NetworkAction(action_id=action_id)

        
        obs = env.step(action)
        
       
        print(f"[STEP] {steps}: action={action_id} reward={obs.reward}")

        history.append({"action": action_id, "reward": obs.reward})
        steps += 1

    
    print(f"[END] {task_name}")
    return history

if __name__ == "__main__":
    print("Running inference with LLM agent...\n")

    all_results = {}

    for task, grader in TASK_GRADERS.items():
        print(f"Running task: {task}")
        history = run_episode(task)
        score = grader(history)
        all_results[task] = round(score, 3)

    print("\nFinal Scores:")
    print(json.dumps(all_results, indent=2))