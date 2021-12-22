FROM python:3.8-slim

COPY server/src/requirements.txt /root/server/src/requirements.txt

RUN chown -R root:root /root/server

WORKDIR /root/server/src
RUN pip3 install --no-cache-dir -r requirements.txt

COPY server/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]
