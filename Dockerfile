# Use the official lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir flask pandas scikit-learn flask-cors gunicorn

# Expose the port for Google Cloud Run
ENV PORT 8080
EXPOSE $PORT

# Run the app using gunicorn (replace app_name with your actual Flask app file name)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
