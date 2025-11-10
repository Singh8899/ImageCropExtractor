# 🚀 Image-Crop-Extractor

This guide walks you through the entire process: from training the model to deploying it on RunPod Serverless.

## 📋 Prerequisites

- **Python 3.11** (managed with pyenv)
- **CUDA GPU** (for training and inference)
- **Hugging Face account** with access token
- **Docker Hub account**
- **RunPod account**
- **Firebase credentials** (`secret.json`)

## 🔧 Initial Setup

### 1. Python Environment Configuration

```bash
# Install Python 3.11 with pyenv (if not already installed)
pyenv install 3.11.9
pyenv activate torch-env
pip install -r requirements.txt
```

### 2. Firebase Configuration

Ensure you have the `secret.json` file with Firebase credentials in the root of the project.

## 🎯 Complete Workflow

### Phase 1: Dataset Preparation

#### Step 1.1: Download Photos from Firebase

```bash
python download_photo.py
```

**What it does**: Downloads images and bounding boxes from Firebase Storage into the following folders:

- `dataset/photo/` - Images
- `dataset/bounding_boxes/` - XML annotations

#### Step 1.2: Generate Dataset Metadata

```bash
python dataset_generator.py
```

**What it does**: Creates the `dataset.json` file containing:

- Image paths
- Training prompts
- Target crop coordinates
- Metadata for fine-tuning

### Phase 2: Model Training

#### Step 2.1: Run Training

```bash
jupyter notebook train.ipynb
python fine_tune_unsloath.py
```

**What happens during training**:

- Loads the base model (Llama-3.2-11B-Vision)
- Fine-tunes it with the dataset
- Saves checkpoints in `outputs_checkpoint/`
- Training may take several hours on a GPU

#### Step 2.2: Monitor Training

Training saves periodic checkpoints in:

```
outputs_checkpoint/
├── checkpoint-100/
├── checkpoint-200/
├── ...
└── checkpoint-950/  # Final checkpoint
```

### Phase 3: Upload Model to Hugging Face

#### Step 3.1: Upload the Trained Model

```bash
python upload_model.py \
  -c "outputs_checkpoint/checkpoint-950" \
  -o "yourusername/model-name" \
  -t "hf_YOUR_HUGGINGFACE_TOKEN"
```

**Parameters**:

- `-c`: Path to the checkpoint to upload
- `-o`: Name of the repository on HF (format: username/model-name)
- `-t`: Hugging Face token

**Example**:

```bash
python upload_model.py \
  -c "outputs_checkpoint/checkpoint-950" \
  -o "gmanuzz/diego_2" \
  -t "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"
```

### Phase 4: Update Worker Code

#### Step 4.1: Update the Model in the Worker

Edit the `workerDiego/model.py` file at line 13:

```python
# Before (old model)
self.model, self.tokenizer = FastVisionModel.from_pretrained(
    "gmanuzz/diego_1", load_in_4bit=False
)

# After (new model)
self.model, self.tokenizer = FastVisionModel.from_pretrained(
    "gmanuzz/diego_2", load_in_4bit=False  # <-- Update here
)
```

### Phase 5: Build and Push Docker

#### Step 5.1: Build the Container

```bash
cd workerDiego
docker build . --tag=yourusername/imagecropextractor:latest
```

#### Step 5.2: Login and Push to Docker Hub

```bash
docker login
docker push yourusername/imagecropextractor:latest
```

**Complete Example**:

```bash
cd workerDiego
docker build . --tag=giuliomanuzzi001/imagecropextractor:latest
docker login
docker push giuliomanuzzi001/imagecropextractor:latest
```

### Phase 6: Deployment on RunPod

#### 🔄 Deployment Type Selection

This project supports two deployment types:

**🎯 Load Balancing (Recommended for production)**

- Always-on HTTP server
- Better performance for frequent requests
- Automatic scaling based on load
- Integrated health checks
- Lower latency

**⚡ Serverless (For occasional use)**

- On-demand activation
- Reduced costs for occasional use
- Slower cold start

---

## 🎯 Load Balancing Deployment (Recommended)

### Step 6A.1: Build Load Balancing Image

```bash
cd workerDiego

# 🎯 Build and deploy Load Balancing
./build_loadbalancer.sh

# Or manually:
docker build -f Dockerfile.loadbalancer -t giuliomanuzzi001/imagecropextractor-loadbalancer:latest .
docker push giuliomanuzzi001/imagecropextractor-loadbalancer:latest
```

### Step 6A.2: Create Load Balancing Endpoint

1. Go to: https://console.runpod.io/endpoints
2. Click "New Endpoint"
3. Select "Load Balancing"
4. Fill in:
   - **Endpoint Name**: `ImageCropExtractor-LoadBalancer`
   - **Docker Image**: `giuliomanuzzi001/imagecropextractor-loadbalancer:latest`
   - **Container Disk**: `24 GB`
   - **Volume Disk**: `60 GB`
   - **Port**: `8000`
   - **GPU Type**: `RTX A4000` or higher
   - **Min Workers**: `1`
   - **Max Workers**: `3`

### Step 6A.3: Test Load Balancing API

```bash
# Test health check
curl https://YOUR_ENDPOINT_ID.api.runpod.ai/ping

# Test prediction
curl -X POST https://YOUR_ENDPOINT_ID.api.runpod.ai/ \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "BASE64_ENCODED_IMAGE_STRING"
    }
  }'
```

**Response format:**

```json
{
  "result": [
    {
      "x1": 100,
      "y1": 50,
      "x2": 300,
      "y2": 250
    }
  ]
}
```

---

## ⚡ Serverless Deployment (Alternative)

### Step 6B.1: Build Serverless Image

```bash
cd workerDiego

# ⚡ Build and deploy Serverless
./build_serverless.sh

# Or manually:
docker build -f Dockerfile -t giuliomanuzzi001/imagecropextractor:latest .
docker push giuliomanuzzi001/imagecropextractor:latest
```

### Step 6B.2: Create Serverless Template

1. Go to: https://console.runpod.io/serverless/user/templates
2. Click "New Template"
3. Fill in:
   - **Template Name**: `ImageCropExtractor-Serverless`
   - **Container Image**: `giuliomanuzzi001/imagecropextractor:latest`
   - **Container Disk**: `24 GB`
   - **Volume Disk**: `60 GB`

### Step 6B.3: Create Serverless API Endpoint

1. Go to: https://console.runpod.io/serverless/user/apis
2. Click "New API"
3. Fill in:
   - **API Name**: `ImageCropExtractor-Serverless`
   - **Select Template**: `ImageCropExtractor-Serverless`
   - **Min Workers**: `0`
   - **Max Workers**: `3`
   - **Idle Timeout**: `5` seconds
   - **Flash Boot**: ✅ Enabled
   - **GPU Type**: `RTX A4000` or higher

### Step 6B.4: Test Serverless API

```bash
# Using Python script (recommended)
python test_serverless_api.py YOUR_ENDPOINT_ID YOUR_RUNPOD_TOKEN

# Using environment variables
export RUNPOD_ENDPOINT_ID=YOUR_ENDPOINT_ID
export RUNPOD_API_TOKEN=YOUR_RUNPOD_TOKEN
python test_serverless_api.py

# Using curl (manual)
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_TOKEN" \
  -d '{
    "input": {
      "image": "BASE64_ENCODED_IMAGE_STRING"
    }
  }'
```

**Serverless response format:**

```json
{
  "id": "sync-abc123",
  "status": "COMPLETED",
  "output": [
    {
      "x1": 100,
      "y1": 50,
      "x2": 300,
      "y2": 250
    }
  ]
}
```
