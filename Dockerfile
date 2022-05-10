FROM python:3.10

RUN apt-get update && apt install -y --no-install-recommends \
    ghostscript

WORKDIR synthetic/
COPY . .

RUN pip install .

ENTRYPOINT [ "synthetic" ]