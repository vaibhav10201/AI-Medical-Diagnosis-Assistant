# AI Medical Diagnosis Assistant 🩺

A Streamlit-based AI Medical Assistant that extracts symptoms from user text, predicts possible diseases using a Logistic Regression classifier, and generates medical information explanations using a custom PyTorch GRU model.

## Run Instructions

1. Activate the virtual environment (if not already activated):
```bash
source .venv/bin/activate
pip install -r requirements.txt && python -m spacy download en_core_web_sm
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

The app will start and automatically open in your default web browser. Note that on the first run, model training and caching may take a minute or two.
