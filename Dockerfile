# --- FILE: Dockerfile ---

# Start from an official Python base image.
# Using a specific version ensures consistency.
FROM python:3.10-slim

# Set the working directory inside the container.
# All subsequent commands will run from this path.
WORKDIR /app

# Copy the requirements file into the container first.
# This allows Docker to cache the dependency installation layer,
# speeding up future builds if the requirements haven't changed.
COPY requirements.txt .

# Install the Python dependencies.
# --no-cache-dir reduces the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container.
# This includes app.py, config.py, and the entire src/ directory.
COPY . .

# Expose the port that Streamlit runs on by default.
# This makes the port available to the host machine.
EXPOSE 8501

# Define the command to run when the container starts.
# This command starts the Streamlit application.
# --server.port allows it to be accessed from outside the container.
# --server.address 0.0.0.0 is necessary to listen on all network interfaces.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]