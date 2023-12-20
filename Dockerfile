# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container's working directory
COPY requirements.txt .

# Install required Python packages based on the requirements.txt file
RUN pip install -r requirements.txt

# Copy the contents of the current directory into the container's working directory (/app)
COPY . .

# Build the model and load necessary data (Assuming app.py has a script to build the model)
RUN python main.py --build-model

# Command to start the application using Gunicorn
CMD exec gunicorn --bind :$PORT --worker 1 --threads 8 --timeout 0 main:app
