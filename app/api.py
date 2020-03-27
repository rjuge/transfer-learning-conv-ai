from fastapi import FastAPI
from pydantic import BaseModel
from model.MyBot import MyBot

MIN_LENGTH = 1
MAX_LENGTH = 20
MAX_HISTORY = 2
TEMPERATURE = 0.7
TOP_K = 60
TOP_P = 0.4
DEVICE = "cpu"

PERSONALITY = [
    "i like to study .",
    "i would like to go to college .",
]

# load bot
bot = MyBot(PERSONALITY)

# initiate API
app = FastAPI()

# define model for post request.
class RequestParams(BaseModel):
    text: str


@app.post("/answer")
def answer(params: RequestParams):
    ans = bot.answer(params.text)
    return ans
