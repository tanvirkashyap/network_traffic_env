from fastapi import FastAPI
from server.environment import NetworkEnvironment
from models import NetworkAction
from pydantic import BaseModel

# request models
class ActionRequest(BaseModel):
    action_id: int

class ResetRequest(BaseModel):
    task_name: str = "obvious"

app = FastAPI()

env = NetworkEnvironment()


@app.post("/reset")
def reset(request: ResetRequest):
    env.task_name = request.task_name
    obs = env.reset()
    return obs.__dict__


@app.post("/step")
def step(action: ActionRequest):
    obs = env.step(NetworkAction(action_id=action.action_id))
    return obs.__dict__


@app.get("/state")
def state():
    return env.state().__dict__