FROM python:3.8-slim

COPY FlaskExample/src/requirements.txt /root/FlaskExample/src/requirements.txt

RUN chown -R root:root /root/FlaskExample

WORKDIR /root/FlaskExample/src
RUN pip3 install -r requirements.txt

COPY FlaskExample/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]