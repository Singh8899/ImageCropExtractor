FROM python:3.11-slim

# Set a proper working directory
WORKDIR /

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rp_handler.py /
COPY model.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]