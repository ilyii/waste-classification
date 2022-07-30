# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /website-docker

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]

CMD flask run -h 0.0.0.0 -p 5000