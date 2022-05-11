FROM python:3.10

RUN apt-get update && apt install -y --no-install-recommends \
    ghostscript

WORKDIR /root/synthetic/
COPY . .
RUN pip install .

WORKDIR /root/synthesizer/

ENV PYTHONPATH "${PYTHONPATH}:/root/synthetic"
ENV PYTHONPATH "${PYTHONPATH}:/root/synthesizer"

ENTRYPOINT [ "synthetic" ]
