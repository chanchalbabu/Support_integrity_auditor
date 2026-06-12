# SIA Deployment Guide
# MARS Open Projects 2026

## 1. LOCAL DEPLOYMENT
```bash
git clone https://github.com/your-username/SIA.git
cd SIA
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

## 2. TRAIN MODEL FIRST
```bash
python train_pipeline.py --skip-advanced   # Fast (2 min)
python train_pipeline.py                   # Full DeBERTa (GPU recommended)
```

## 3. DOCKER DEPLOYMENT
```bash
docker-compose up --build
# → http://localhost:8501
# Train: docker-compose --profile train up sia-train
```

## 4. STREAMLIT CLOUD
1. Push repo to GitHub (public)
2. Visit https://share.streamlit.io
3. New app → your repo → app.py → Python 3.11
4. Done — free hosting, auto-deploys on git push

## 5. HUGGING FACE SPACES
1. Create Space at huggingface.co/spaces
2. SDK: Streamlit | Hardware: CPU Basic (free)
3. Upload all files or connect GitHub repo
4. Add requirements.txt → auto-builds

## 6. RENDER
1. New Web Service → connect GitHub
2. Build: pip install -r requirements.txt
3. Start: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
4. Deploy → free tier available

## ENVIRONMENT VARIABLES (optional)
```
PYTHONPATH=/app
STREAMLIT_THEME_BASE=light
```

## VERIFY DEPLOYMENT
```bash
curl http://localhost:8501/_stcore/health
# Should return: {"status": "ok"}
```
