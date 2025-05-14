FROM python:3.11-slim

WORKDIR /app

# Installa le dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia i file di configurazione delle dipendenze
COPY pyproject.toml uv.lock ./

# Copia il codice dell'applicazione
COPY . .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt --upgrade

# Crea la directory per gli upload e tmp se non esistono
#RUN mkdir -p uploads
RUN mkdir -p /tmp/torch_shm

# Espone la porta su cui l'applicazione si avvierà
EXPOSE 5000

# Configurazione ambiente per PyTorch
ENV OMP_NUM_THREADS=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV TORCH_DISTRIBUTED_DEBUG=DETAIL

# Comando per avviare l'applicazione
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "--workers", "2", "--timeout", "900", "main:app"]