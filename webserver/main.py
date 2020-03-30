import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import redis
import json
import time

# initiate API
app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))

# define model for post request.
class RequestParams(BaseModel):
    text: str


@app.post("/answer")
def answer(params: RequestParams):
    data = {"success": False}
    # Generate an ID for the message then add the ID + message to the queue
    k = str(uuid.uuid4())
    d = {"id": k, "message": params.text}
    db.rpush(os.environ.get("MESSAGE_QUEUE"), json.dumps(d))

    # Keep looping for CLIENT_MAX_TRIES times
    num_tries = 0
    while num_tries < CLIENT_MAX_TRIES:
        num_tries += 1

        # Attempt to grab the output predictions
        output = db.get(k)

        # Check to see if our model has answered
        if output is not None:
            # Add the output predictions to our data dictionary so we can return it to the client
            output = output.decode("utf-8")
            data["predictions"] = json.loads(output)

            # Delete the result from the database and break from the polling loop
            db.delete(k)
            break

        # Sleep for a small amount to give the model a chance to classify the input image
        time.sleep(float(os.environ.get("CLIENT_SLEEP")))

        # Indicate that the request was a success
        data["success"] = True
    else:
        raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
    return data
