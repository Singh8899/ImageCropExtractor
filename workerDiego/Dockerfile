FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Fix per bitsandbytes e CUDA
ENV BNB_CUDA_VERSION=124
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Set a proper working directory
WORKDIR /

# Copy requirements file
COPY requirements.txt /

# Fix per bitsandbytes - reinstalla con CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y bitsandbytes && \
    pip install --no-cache-dir bitsandbytes --no-deps && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY rp_handler_serverless.py /
COPY model.py /

# Start the container
CMD ["python3", "-u", "rp_handler_serverless.py"]