# 🚀 ImageCropExtractor

Questa guida ti accompagna attraverso tutto il processo: dal training del
modello fino al deployment su RunPod Serverless.

## 📋 Prerequisiti

- **Python 3.11** (gestito con pyenv)
- **CUDA GPU** (per training e inference)
- **Account Hugging Face** con token di accesso
- **Account Docker Hub**
- **Account RunPod**
- **Firebase credentials** (`secret.json`)

## 🔧 Setup Iniziale

### 1. Configurazione Ambiente Python

```bash
# Installa Python 3.11 con pyenv (se non già presente)
pyenv install 3.11.9
pyenv activate torch-env
pip install -r requirements.txt
```

### 2. Configurazione Firebase

Assicurati di avere il file `secret.json` con le credenziali Firebase nella root
del progetto.

## 🎯 Workflow Completo

### Fase 1: Preparazione Dataset

#### Step 1.1: Download delle Foto da Firebase

```bash
python download_photo.py
```

**Cosa fa**: Scarica le immagini e i bounding boxes dal Firebase Storage nelle
cartelle:

- `dataset/photo/` - Immagini
- `dataset/bounding_boxes/` - Annotazioni XML

#### Step 1.2: Generazione Metadata Dataset

```bash
python dataset_generator.py
```

**Cosa fa**: Crea il file `dataset.json` che contiene:

- Percorsi delle immagini
- Prompt di training
- Coordinate dei crop target
- Metadati per il fine-tuning

### Fase 2: Training del Modello

#### Step 2.1: Esecuzione Training

```bash
jupyter notebook train.ipynb
python fine_tune_unsloath.py
```

**Cosa succede durante il training**:

- Caricamento del modello base (Llama-3.2-11B-Vision)
- Fine-tuning con i dati del dataset
- Salvataggio checkpoint in `outputs_checkpoint/`
- Il training può richiedere diverse ore su GPU

#### Step 2.2: Monitoraggio Training

Il training salva checkpoint periodici in:

```
outputs_checkpoint/
├── checkpoint-100/
├── checkpoint-200/
├── ...
└── checkpoint-950/  # Checkpoint finale
```

### Fase 3: Upload Modello su Hugging Face

#### Step 3.1: Upload del Modello Trainato

```bash
python upload_model.py \
  -c "outputs_checkpoint/checkpoint-950" \
  -o "tuousername/nome-modello" \
  -t "hf_TUO_TOKEN_HUGGINGFACE"
```

**Parametri**:

- `-c`: Path al checkpoint da uploadare
- `-o`: Nome del repository su HF (formato: username/model-name)
- `-t`: Token Hugging Face

**Esempio**:

```bash
python upload_model.py \
  -c "outputs_checkpoint/checkpoint-950" \
  -o "gmanuzz/diego_2" \
  -t "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"
```

### Fase 4: Aggiornamento Codice Worker

#### Step 4.1: Aggiorna il Modello nel Worker

Modifica il file `workerDiego/model.py` alla riga 13:

```python
# Prima (modello vecchio)
self.model, self.tokenizer = FastVisionModel.from_pretrained(
    "gmanuzz/diego_1", load_in_4bit=False
)

# Dopo (nuovo modello)
self.model, self.tokenizer = FastVisionModel.from_pretrained(
    "gmanuzz/diego_2", load_in_4bit=False  # <-- Cambia qui
)
```

### Fase 5: Build e Push Docker

#### Step 5.1: Build del Container

```bash
cd workerDiego
docker build . --tag=tuousername/imagecropextractor:latest
```

#### Step 5.2: Login e Push su Docker Hub

```bash
docker login
docker push tuousername/imagecropextractor:latest
```

**Esempio completo**:

```bash
cd workerDiego
docker build . --tag=giuliomanuzzi001/imagecropextractor:latest
docker login
docker push giuliomanuzzi001/imagecropextractor:latest
```

### Fase 6: Deployment su RunPod

#### 🔄 Scelta del Tipo di Deployment

Questo progetto supporta due tipi di deployment:

**🎯 Load Balancing (Raccomandato per produzione)**

- Server HTTP sempre attivo
- Migliori performance per richieste frequenti
- Scaling automatico basato sul carico
- Health checks integrati
- Latenza più bassa

**⚡ Serverless (Per uso sporadico)**

- Attivazione on-demand
- Costi ridotti per uso occasionale
- Cold start più lento

---

## 🎯 Deployment Load Balancing (Raccomandato)

### Step 6A.1: Build Immagine Load Balancing

```bash
cd workerDiego

# 🎯 Build e deploy Load Balancing
./build_loadbalancer.sh

# Oppure manualmente:
docker build -f Dockerfile.loadbalancer -t giuliomanuzzi001/imagecropextractor-loadbalancer:latest .
docker push giuliomanuzzi001/imagecropextractor-loadbalancer:latest
```

### Step 6A.2: Crea Load Balancing Endpoint

1. Vai su: https://console.runpod.io/endpoints
2. Clicca "New Endpoint"
3. Seleziona "Load Balancing"
4. Compila:
   - **Endpoint Name**: `ImageCropExtractor-LoadBalancer`
   - **Docker Image**: `giuliomanuzzi001/imagecropextractor-loadbalancer:latest`
   - **Container Disk**: `24 GB`
   - **Volume Disk**: `60 GB`
   - **Port**: `8000`
   - **GPU Type**: `RTX A4000` o superiore
   - **Min Workers**: `1`
   - **Max Workers**: `3`

### Step 6A.3: Test Load Balancing API

```bash
# Test health check
curl https://TUO_ENDPOINT_ID.api.runpod.ai/ping

# Test predizione
curl -X POST https://TUO_ENDPOINT_ID.api.runpod.ai/ \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "BASE64_ENCODED_IMAGE_STRING"
    }
  }'
```

**Formato risposta:**

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

## ⚡ Deployment Serverless (Alternativo)

### Step 6B.1: Build Immagine Serverless

```bash
cd workerDiego

# ⚡ Build e deploy Serverless
./build_serverless.sh

# Oppure manualmente:
docker build -f Dockerfile -t giuliomanuzzi001/imagecropextractor:latest .
docker push giuliomanuzzi001/imagecropextractor:latest
```

### Step 6B.2: Crea Template Serverless

1. Vai su: https://console.runpod.io/serverless/user/templates
2. Clicca "New Template"
3. Compila:
   - **Template Name**: `ImageCropExtractor-Serverless`
   - **Container Image**: `giuliomanuzzi001/imagecropextractor:latest`
   - **Container Disk**: `24 GB`
   - **Volume Disk**: `60 GB`

### Step 6B.3: Crea API Endpoint Serverless

1. Vai su: https://console.runpod.io/serverless/user/apis
2. Clicca "New API"
3. Compila:
   - **API Name**: `ImageCropExtractor-Serverless`
   - **Select Template**: `ImageCropExtractor-Serverless`
   - **Min Workers**: `0`
   - **Max Workers**: `3`
   - **Idle Timeout**: `5` secondi
   - **Flash Boot**: ✅ Abilitato
   - **GPU Type**: `RTX A4000` o superiore

### Step 6B.4: Test Serverless API

```bash
# Con script Python (raccomandato)
python test_serverless_api.py TUO_ENDPOINT_ID TUO_RUNPOD_TOKEN

# Con variabili d'ambiente
export RUNPOD_ENDPOINT_ID=TUO_ENDPOINT_ID
export RUNPOD_API_TOKEN=TUO_RUNPOD_TOKEN
python test_serverless_api.py

# Con curl (manuale)
curl -X POST https://api.runpod.ai/v2/TUO_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TUO_RUNPOD_TOKEN" \
  -d '{
    "input": {
      "image": "BASE64_ENCODED_IMAGE_STRING"
    }
  }'
```

**Formato risposta serverless:**

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
