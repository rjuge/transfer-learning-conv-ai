FROM tiangolo/uvicorn-gunicorn:python3.6

COPY . /app
COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
