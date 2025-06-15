
# 📦 Dockerfile

FROM python:3.9-slim

# Set workdir inside the container
WORKDIR /app

# Copy local files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask pandas scikit-learn

# Expose the port Flask runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
