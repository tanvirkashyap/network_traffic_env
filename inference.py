import os
import json
from openai import OpenAI
from client import NetworkEnvClient
from models import NetworkAction
from tasks.task1_obvious import grader as grader1
from tasks.task2_subtle import grader as grader2
from tasks.task3_mixed import grader as grader3
# MANDATORY: judges provide these. 
# For your local testing, we point to Hugging Face's Brain
LLM_API_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Your Environment URL
ENV_URL = os.getenv("ENV_URL", "https://tanananana-network-traffic-env.hf.space")

openai_client = OpenAI(
    base_url=LLM_API_URL, 
    api_key=HF_TOKEN 
)
env = NetworkEnvClient(base_url=ENV_URL)


TASK_GRADERS = {
    "obvious": grader1,
    "subtle": grader2,
    "mixed": grader3
}

def choose_action(obs):
    prompt = f"""
You are a cybersecurity analyst monitoring network traffic.
Decide what to do with this connection:

- Duration: {obs['duration']}s
- Source bytes: {obs['src_bytes']}
- Destination bytes: {obs['dst_bytes']}
- Wrong fragments: {obs['wrong_fragment']}
- Failed logins: {obs['num_failed_logins']}
- Recent attacks in window: {obs['window_attack_count']}
- Similar connections in window: {obs['window_same_src_count']}

Actions:
0 = allow (safe connection)
1 = flag (suspicious, needs review)
2 = block (malicious, block immediately)

Respond ONLY with 0, 1, or 2.
"""
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    action_text = response.choices[0].message.content.strip()
    try:
        return int(action_text)
    except:
        return 1  # fallback to flag, safest default

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