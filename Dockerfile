FROM python:3.8.6 as base

FROM base as train

ARG DVC_SSH_KEY
ARG DVC_SSH_CONFIG
RUN mkdir /root/.ssh/
RUN echo "${DVC_SSH_KEY}" > /root/.ssh/id_rsa
RUN echo "${DVC_SSH_CONFIG}" > /root/.ssh/config

COPY . .
RUN pip install -r requirements.txt
RUN dvc pull --run-cache
RUN dvc repro
RUN dvc push --run-cache

FROM base
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY --from=train models/ models/
COPY src/serve.py src/serve.py

EXPOSE 80

ENTRYPOINT [ "python" ]
CMD [ "src/serve.py" ]
