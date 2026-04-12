from urllib import request

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
def reset(request: ResetRequest):
    env.task_name = request.task_name
    obs = env.reset()
    return {
        "observation": obs.__dict__,
        "reward": None,
        "done": False,
        "info": {}
    }

@app.post("/step")
def step(action: ActionRequest):
    obs = env.step(NetworkAction(action_id=action.action_id))
    return {
        "observation": obs.__dict__,
        "reward": obs.reward,
        "done": obs.done,
        "info": {}
    }


@app.get("/state")
def state():
    return env.state().dict()

def main():
    """Main entry point for the OpenEnv validator."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()