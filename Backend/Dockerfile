# This will be the image. Can be found on docker website. This installs python
FROM python:3.10.6-buster

# Set the working directory to /app
WORKDIR /app

# Copy all files necessary for you to use on your docker image. Like installing something on your own laptop
COPY app /app

COPY requirements.txt /app/requirements.txt

COPY model_test2 /app/model_test2

# You use the run to install dependencies. This will install it into the docker image
RUN pip install --upgrade pip

RUN pip install -r /app/requirements.txt

# You need to tell docker that it needs to start the uvicorn server
# app.simple:app will be broken down as app(folder).simple(py file):app(what api was called inside py file app = FastAPI())
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
