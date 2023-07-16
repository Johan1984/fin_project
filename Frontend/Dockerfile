FROM python:3.10.6-buster

WORKDIR /app

EXPOSE 8080

COPY requirements.txt .

# You use the run to install dependencies. This will install it into docker image

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
