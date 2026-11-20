# RogMitra.AI - CPU-only image for the Streamlit app.
# Build:  docker build -t rogmitra-ai .
# Run:    docker run --rm -p 8501:7860 rogmitra-ai      ->  http://localhost:8501
# The same image works on Hugging Face Spaces (Docker SDK), which expects port 7860.
FROM python:3.11-slim

# Why these flags: no .pyc clutter, unbuffered logs, and no pip cache so the image stays small.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Hugging Face runs containers as user 1000, so create that user and work in its home.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install dependencies BEFORE copying the code, so editing code does not re-download everything.
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# Code + trained models + processed data + reports (see .dockerignore for what is excluded).
COPY --chown=user:user . .
USER user

ENV PORT=7860
EXPOSE 7860

# Plain-Python health check (the slim image has no curl).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT','7860'), timeout=4)"

CMD streamlit run app/Home.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false
