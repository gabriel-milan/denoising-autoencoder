FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
COPY ./app /app
RUN pip3 install --no-cache-dir -U -r /app/requirements.txt
