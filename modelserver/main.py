import time
import json
import redis

from MyBot import MyBot

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

# Connect to Redis server
db = redis.StrictRedis(host="redis", port=6379)


def run():

    bot = MyBot(PERSONALITY)

    # Continually poll for new messages
    while True:
        # Pop off first message
        q = db.lpop("message_queue")
        if q is not None:
            d = json.loads(q.decode("utf-8"))
            ans = bot.answer(d["message"])
            output = {"answer": ans}
            db.set(d["id"], json.dumps(output))

            # Sleep for a small amount
            time.sleep(0.2)


if __name__ == "__main__":
    run()
