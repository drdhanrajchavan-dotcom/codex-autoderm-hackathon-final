FROM node:22-bookworm-slim

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_PUBLIC_API_BASE_URL="" \
    AUTODERM_MAX_UPLOAD_BYTES=8388608 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/app/node_modules/.bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY .npmrc package.json package-lock.json ./
COPY web/package.json ./web/package.json
RUN npm ci --include=dev

COPY requirements.txt ./
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p config data/cropped \
    && printf '{\n  "weights_path": ".pulled_artifacts/runs/iter_018/weights/best.pt",\n  "preprocessing": "cropped",\n  "iteration_id": "iter_018"\n}\n' > config/active_checkpoint.json \
    && cp .pulled_artifacts/runs/iter_018/weights/preprocessing_hash.txt data/cropped/preprocessing_hash.txt \
    && chmod +x scripts/start_prod.sh \
    && npm run build

EXPOSE 3000

CMD ["./scripts/start_prod.sh"]
