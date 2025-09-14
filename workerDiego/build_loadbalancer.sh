#!/bin/bash

# Script per buildare e deployare l'immagine Docker per RunPod Load Balancing

set -e  # Esci se qualsiasi comando fallisce

# Configurazione
DOCKER_USERNAME="giuliomanuzzi001"
IMAGE_NAME="imagecropextractor-loadbalancer"
TAG="v2-fixed"
FULL_IMAGE_NAME="$DOCKER_USERNAME/$IMAGE_NAME:$TAG"

echo "🎯 Build e Deploy Load Balancing per RunPod"
echo "==========================================="
echo "Immagine: $FULL_IMAGE_NAME"
echo ""

# Controlla se Docker è in esecuzione
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker non è in esecuzione. Avvia Docker Desktop e riprova."
    exit 1
fi

# Controlla se siamo nella directory corretta
if [ ! -f "Dockerfile.loadbalancer" ]; then
    echo "❌ File Dockerfile.loadbalancer non trovato. Assicurati di essere nella directory workerDiego."
    exit 1
fi

# Build dell'immagine
echo "🔨 Building Docker image..."
docker build -f Dockerfile.loadbalancer -t $FULL_IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "✅ Build completato con successo!"
else
    echo "❌ Build fallito!"
    exit 1
fi

# Mostra informazioni sull'immagine
echo ""
echo "📊 Informazioni immagine:"
docker images $FULL_IMAGE_NAME

# Chiedi conferma per il push
echo ""
read -p "🚢 Vuoi pushare l'immagine su Docker Hub? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚢 Pushing to Docker Hub..."
    
    # Login a Docker Hub (se necessario)
    echo "🔐 Effettua il login a Docker Hub se richiesto:"
    docker login
    
    # Push dell'immagine
    docker push $FULL_IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        echo "✅ Push completato con successo!"
        echo ""
        echo "🎯 Immagine disponibile su: https://hub.docker.com/r/$DOCKER_USERNAME/$IMAGE_NAME"
        echo ""
        echo "📝 Per usare su RunPod:"
        echo "   Docker Image: $FULL_IMAGE_NAME"
        echo "   Port: 8000"
        echo "   Health Check: /ping"
        echo ""
    else
        echo "❌ Push fallito!"
        exit 1
    fi
else
    echo "⏭️  Push saltato. Immagine disponibile solo localmente."
fi

echo ""
echo "🎉 Processo completato!"
echo ""
echo "📋 Prossimi passi:"
echo "1. Vai su RunPod Console"
echo "2. Aggiorna il tuo Load Balancing endpoint"
echo "3. Cambia Docker Image in: $FULL_IMAGE_NAME"
echo "4. Assicurati che la porta sia impostata su 8000"
echo "5. Testa l'endpoint con una richiesta POST"
