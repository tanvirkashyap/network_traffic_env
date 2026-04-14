import os
import json
from openai import OpenAI
from client import NetworkEnvClient
from models import NetworkAction

# 1. IMPORT GRADERS
from tasks.task1_obvious import grader as grader1
from tasks.task2_subtle import grader as grader2
from tasks.task3_mixed import grader as grader3


LLM_API_URL = os.getenv("API_BASE_URL")
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
ENV_URL = os.getenv("ENV_URL", "https://tanananana-network-traffic-env.hf.space")

openai_client = OpenAI(base_url=LLM_API_URL, api_key=os.getenv("API_KEY"))
env = NetworkEnvClient(base_url=ENV_URL)

TASK_GRADERS = {
    "obvious": grader1,
    "subtle": grader2,
    "mixed": grader3
}

# ==========================================================
# LLM LOGIC
# ==========================================================
def choose_action(obs_dict: dict) -> int:
    # convert any numpy types to plain Python
    safe_obs = {k: (int(v) if hasattr(v, 'item') else v) 
                for k, v in obs_dict.items()}
    
    messages = [
        {
            "role": "system",
            "content": "You are a cybersecurity firewall. Respond with ONLY a single digit: 0, 1, or 2. No explanation."
        },
        {
            "role": "user",
           "content": f"""Analyze this network connection:

 RECENT ATTACKS IN WINDOW: {safe_obs['window_attack_count']} (if > 0, be suspicious)

- src_bytes: {safe_obs['src_bytes']}
- duration: {safe_obs['duration']}s
- wrong_fragment: {safe_obs['wrong_fragment']}
- num_failed_logins: {safe_obs['num_failed_logins']}
- window_same_src_count: {safe_obs['window_same_src_count']}

IMPORTANT: Missing an attack is worse than a false positive.
When in doubt, flag or block.

0 = allow (you are confident this is safe)
1 = flag (suspicious)
2 = block (clearly malicious)

Respond with ONLY 0, 1, or 2."""
        }
    ]

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        #print(f"DEBUG model output: '{content}'"), checking if the parsing is working correctly
        
        # check last character first
        if content and content[-1] in ["0", "1", "2"]:
            return int(content[-1])
        
        # scan whole response
        for char in content:
            if char in ["0", "1", "2"]:
                return int(char)
        
        return 1  # fallback
        
    except Exception as e:
        print(f"❌ LLM ERROR: {e}")
        return 1
# ==========================================================
# EPISODE LOOP
# ==========================================================
def run_episode(task_name="obvious", max_steps=100):
    """
    Runs an episode and prints mandatory logs in [START], [STEP], [END] format.
    """
    obs = env.reset(task_name=task_name)
    history = []
    
    # MANDATORY FORMAT
    print(f"[START] {task_name}")

    steps = 0
    while not obs.done and steps < max_steps:
        action_id = choose_action(obs.__dict__)
        action = NetworkAction(action_id=action_id)
    
        history.append({"action": action_id, "reward": obs.reward if obs.reward is not None else 0.0})
    
        obs = env.step(action)
        print(f"[STEP] {steps}: action={action_id} reward={obs.reward}")
        steps += 1
    
    # MANDATORY FORMAT
    print(f"[END] {task_name}")
    return history

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN is not set in terminal.")
    else:
        #print(f"📡 Brain API: {LLM_API_URL}")
        #print(f"🤖 Brain Model: {MODEL_NAME}")
        #print(f"🛡️ Env URL: {ENV_URL}\n")

        all_results = {}

        for task, grader_func in TASK_GRADERS.items():
            # Run task
            history = run_episode(task)
            
            # Grade task
            score = grader_func(history)
            all_results[task] = round(score, 3)

        print("\n=========================")
        print("FINAL EVALUATION SUMMARY:")
        print(json.dumps(all_results, indent=2))
        print("=========================")