FROM hunyuanvideo/hunyuanvideo:cuda_12

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/HunyuanVideo

RUN pip install --no-cache-dir runpod==1.7.9

COPY HunyuanVideo/ /workspace/HunyuanVideo/
RUN pip install --no-cache-dir -r /workspace/HunyuanVideo/requirements.txt

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
