import os
import json
from openai import OpenAI
from client import NetworkEnvClient
from models import NetworkAction

# 1. IMPORT GRADERS
from tasks.task1_obvious import grader as grader1
from tasks.task2_subtle import grader as grader2
from tasks.task3_mixed import grader as grader3

LLM_API_URL = "https://router.huggingface.co/hf-inference/v1"
HF_TOKEN = os.getenv("HF_TOKEN")

# Mistral 7B v0.3 is the most reliable "always-on" model for this API
MODEL_ID =  "meta-llama/Llama-3.1-8B-Instruct"

# 2. THE ENVIRONMENT (Your Space)
ENV_URL = os.getenv("ENV_URL", "https://tanananana-network-traffic-env.hf.space")

# 3. INITIALIZE CLIENTS
# Ensure there is no trailing slash after /v1
openai_client = OpenAI(
    base_url=LLM_API_URL, 
    api_key=HF_TOKEN
)
env = NetworkEnvClient(base_url=ENV_URL)

print(f"📡 Connecting to Brain at: {LLM_API_URL}")
print(f"🤖 Using Model: {MODEL_ID}")

TASK_GRADERS = {
    "obvious": grader1,
    "subtle": grader2,
    "mixed": grader3
}

def choose_action(obs_dict):
    """
    Sends the observation to the LLM and extracts the action ID.
    """
    prompt = f"""[SYSTEM]: You are a cybersecurity analyst. 
Respond ONLY with a single digit: 0, 1, or 2.

[TRAFFIC DATA]:
- Duration: {obs_dict['duration']}s
- Protocol/Service: {obs_dict['protocol_type']} / {obs_dict['service']}
- Bytes (Src/Dst): {obs_dict['src_bytes']} / {obs_dict['dst_bytes']}
- Window Attack Count: {obs_dict['window_attack_count']}

Action (0=Allow, 1=Flag, 2=Block):"""

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        
        # Robust extraction: Look for the first 0, 1, or 2 in the response
        for char in content:
            if char in ["0", "1", "2"]:
                return int(char)
        
        return 1  # Fallback to 'Flag' if no digit found
    except Exception as e:
        print(f"LLM Error: {e}")
        return 1  # Fallback to 'Flag' on connection error

def run_episode(task_name="obvious", max_steps=100):
    """
    Runs a single episode and prints mandatory OpenEnv logs.
    """
    obs = env.reset(task_name=task_name)
    history = []
    
    # MANDATORY LOGGING FORMAT
    print(f"[START] {task_name}")

    steps = 0
    while not obs.done and steps < max_steps:
        # Get action from LLM (using model_dump() as required for Pydantic V2)
        action_id = choose_action(obs.model_dump())
        action = NetworkAction(action_id=action_id)
        
        # Take step
        obs = env.step(action)
        
        # MANDATORY LOGGING FORMAT
        print(f"[STEP] {steps}: action={action_id} reward={obs.reward}")
        
        history.append({"action": action_id, "reward": obs.reward})
        steps += 1
    
    # MANDATORY LOGGING FORMAT
    print(f"[END] {task_name}")
    return history

if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN is not set. Run 'set HF_TOKEN=...' in your terminal.")
    else:
        print(f"🚀 Environment URL: {ENV_URL}")
        print(f"🚀 LLM Model: {MODEL_ID}")
        print("Starting Inference...\n")

        all_results = {}

        for task, grader_func in TASK_GRADERS.items():
            # Run the episode
            history = run_episode(task)
            
            # Grade the performance
            score = grader_func(history)
            all_results[task] = round(score, 3)

        print("\n=========================")
        print("FINAL EVALUATION SUMMARY:")
        print(json.dumps(all_results, indent=2))
        print("=========================")