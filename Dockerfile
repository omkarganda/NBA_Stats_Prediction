FROM python:3.11-slim

# Install system dependencies (optional, but useful for many data/science libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory inside the container
WORKDIR /app

# Copy dependency list and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit
# If your main file is NOT app.py, change "app.py" below accordingly
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
