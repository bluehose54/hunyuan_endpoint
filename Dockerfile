FROM hunyuanvideo/hunyuanvideo:cuda_12

WORKDIR /workspace

RUN pip install --no-cache-dir runpod==1.7.9

COPY HunyuanVideo/ /workspace/HunyuanVideo/
RUN pip install --no-cache-dir -r /workspace/HunyuanVideo/requirements.txt

COPY handler.py /workspace/handler.py

ENV PYTHONPATH=/workspace/HunyuanVideo

CMD ["python3", "-u", "/workspace/handler.py"]
