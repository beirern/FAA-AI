FROM python:3.13-slim-bookworm

WORKDIR /ai

COPY ./requirements.txt /ai/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /ai/requirements.txt

COPY ./main.py /ai/main.py

COPY ./static /ai/static

RUN touch /ai/__init__.py

COPY ./app /ai/app