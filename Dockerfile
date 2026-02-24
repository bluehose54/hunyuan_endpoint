FROM hunyuanvideo/hunyuanvideo:cuda_12

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/HunyuanVideo

RUN pip install --no-cache-dir runpod==1.7.9

# Bring in the repo code (no weights)
COPY HunyuanVideo/ /workspace/HunyuanVideo/

# Install repo deps first
RUN pip install --no-cache-dir -r /workspace/HunyuanVideo/requirements.txt

# Fix optree warning
RUN pip install --no-cache-dir --upgrade "optree>=0.13.0"

# Force compatible versions for HunyuanVideo (last write wins)
RUN pip install --no-cache-dir --upgrade \
    diffusers==0.31.0 \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    accelerate==1.1.1 \
    einops==0.7.0

# Reinstall torchvision to match torch 2.4.0 + CUDA 12.4 (fixes torchvision::nms)
RUN pip install --no-cache-dir --upgrade --force-reinstall \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Only needed if your patched save function uses imageio (keep if unsure)
RUN pip install --no-cache-dir imageio imageio-ffmpeg

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
