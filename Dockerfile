FROM python:3.10-slim

ENV MONGO_DB_USERNAME=admin \
    MONGO_DB_PWD=qwerty

RUN mkdir -p /Spam-detector

COPY . /Spam-detector

WORKDIR /Spam-detector

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "./app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
