FROM python:3.14-slim

ARG WITH_TORCH=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /srv

COPY pyproject.toml ./
COPY bearing ./bearing
# CPU-only PyTorch, and only when serving the CNN: it adds ~700 MB to the image.
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && pip install .

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8100
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/livez')"
CMD ["uvicorn", "bearing.serve.app:app", "--host", "0.0.0.0", "--port", "8100"]
