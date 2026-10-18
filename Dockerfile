FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first, so easyocr doesn't pull the multi-GB CUDA build
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

RUN useradd --create-home --uid 1000 bot \
    && mkdir -p /app/data \
    && chown bot:bot /app/data
USER bot

# Bake EasyOCR models into the image instead of downloading on every start
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY --chown=bot:bot main.py .
COPY --chown=bot:bot src/ src/

CMD ["python", "main.py"]
