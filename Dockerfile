# Server image: the Streamlit demo with the full torch/ultralytics pipeline.
#   docker build -t bovid .
#   docker run -p 8501:8501 bovid            # open http://localhost:8501
#
# Python 3.12: TensorFlow is not needed to serve (the server path is torch), and 3.12 has the
# widest wheel coverage. Dependency ranges resolve for the linux/x86 platform rather than forcing
# the macOS-frozen lock; ultralytics stays pinned via pyproject.toml.
FROM python:3.12-slim

# opencv-python needs libGL + glib at runtime; slim images ship neither.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch: the default Linux torch wheel drags in the multi-GB CUDA stack this CPU server
# never uses. Install it from the CPU index first so the package resolve treats it as satisfied.
COPY pyproject.toml README.md ./
COPY bovid ./bovid
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir ".[server,app]"

# Weights and the small test split (for the app's example images). The full dataset is excluded
# via .dockerignore; only the 33-image test split is copied.
COPY models/best/best.pt models/best/best.pt
COPY models/pretrained/yolov8x.pt models/pretrained/yolov8x.pt
COPY models/pretrained/yolov8s-cls.pt models/pretrained/yolov8s-cls.pt
COPY datasets/indianbovine1/test datasets/indianbovine1/test
COPY app.py ./

ENV BOVID_DEVICE=cpu
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
