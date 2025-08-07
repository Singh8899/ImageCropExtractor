FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set a proper working directory
WORKDIR /

# Copy requirements file
COPY requirements.txt /

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rp_handler.py /
COPY model.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]