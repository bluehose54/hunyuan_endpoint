FROM hunyuanvideo/hunyuanvideo:cuda_12

WORKDIR /workspace

RUN pip install --no-cache-dir runpod==1.7.9

COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
