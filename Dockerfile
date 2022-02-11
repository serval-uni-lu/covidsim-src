FROM python:3.8.6 as base

COPY . .

FROM base
COPY requirements.txt .
RUN pip install -r requirements.txt
