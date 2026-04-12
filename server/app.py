from fastapi import FastAPI
from server.environment import NetworkEnvironment
from models import NetworkAction
from pydantic import BaseModel
import uvicorn

# request models
class ActionRequest(BaseModel):
    action_id: int

class ResetRequest(BaseModel):
    task_name: str = "obvious"

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

env = NetworkEnvironment()


@app.post("/reset")
def reset(request: ResetRequest = None): # <--- Add '= None' here
    # If the validator sends no body, we create a default ResetRequest
    if request is None:
        request = ResetRequest(task_name="obvious")
        
    env.task_name = request.task_name
    obs = env.reset()
    
    # Ensure you are returning the NESTED structure the validator wants
    return {
        "observation": obs.model_dump(),
        "reward": None,
        "done": False,
        "info": {}
    }

@app.post("/step")
def step(action_req: ActionRequest):
    # Process the action through your environment
    obs = env.step(NetworkAction(action_id=action_req.action_id))
    
    # Return the EXACT nested structure required by OpenEnv
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": {"step": obs.step}
    }


@app.get("/state")
def state():
    return env.state().dict()

def main():
    """Main entry point for the OpenEnv validator."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()