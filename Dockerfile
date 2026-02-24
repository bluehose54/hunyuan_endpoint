FROM hunyuanvideo/hunyuanvideo:cuda_12

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/HunyuanVideo

# Install runpod
RUN pip install --no-cache-dir runpod==1.7.9

RUN python3 -m pip install --no-cache-dir --upgrade --force-reinstall \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip cache purge || true

# Copy requirements first for layer caching
COPY HunyuanVideo/requirements.txt /workspace/requirements.txt

# Install repo deps (without torch pin)
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Fix optree
RUN pip install --no-cache-dir --upgrade "optree>=0.13.0"

# Force compatible stack
RUN pip install --no-cache-dir --upgrade \
    diffusers==0.31.0 \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    accelerate==1.1.1 \
    einops==0.7.0
# Optional: video saving
RUN pip install --no-cache-dir imageio imageio-ffmpeg

# Copy full repo
COPY HunyuanVideo/ /workspace/HunyuanVideo/
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
