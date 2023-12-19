FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Build the model and load necessary data
RUN python app.py --build-model

CMD exec gunicorn --bind :$PORT --worker 1 --threads 8 --timeout 0 app:app
