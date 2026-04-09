# Medical Food Recommendation App — Phase 1 + Phase 2
# Complete project README

## Folder Structure

```
phase1/
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← Day 1-2: Understand all 4 datasets
│   ├── 02_data_preprocessing.ipynb    ← Day 7-8: Merge + clean ML training data
│   └── 03_integration_test.ipynb      ← Day 7-8: Test all 3 input modes end-to-end
│
├── backend/
│   └── app/
│       ├── services/
│       │   ├── medical_rules.py       ← Day 3-4: Rule engine (CORE logic)
│       │   ├── food_lookup.py         ← Day 5-6: Fuzzy food search
│       │   └── ocr_pipeline.py        ← Day 15-18: OCR text extraction
│       └── data/
│           ├── food_db_final_.csv              ← COPY HERE
│           ├── Personalized_Diet_Recommendations.csv  ← COPY HERE
│           ├── detailed_meals_macros_CLEANED.csv      ← COPY HERE
│           └── Food_and_Nutrition__.csv               ← COPY HERE
│
└── requirements_phase1.txt
```

## Setup (Day 1)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements_phase1.txt

# 3. Copy your CSV files to backend/app/data/

# 4. Install Tesseract (for OCR - Day 15)
# Linux:   sudo apt install tesseract-ocr tesseract-ocr-eng
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 5. Launch Jupyter
jupyter notebook
```

## Run Order

1. `01_data_exploration.ipynb`   — understand the data (Day 1-2)
2. `02_data_preprocessing.ipynb` — clean + merge training data (Day 7-8)
3. `03_integration_test.ipynb`   — verify everything works (Day 7-8)

## Self-test any .py file directly

```bash
cd backend
python -m app.services.medical_rules   # runs if __name__ == '__main__' block
python -m app.services.food_lookup
python -m app.services.ocr_pipeline
```

## Key outputs after Phase 1

| File | Used by |
|------|---------|
| `backend/app/data/merged_training_data.csv` | Phase 2 ML training |
| `backend/app/data/feature_columns.json` | Phase 2 ML model |
| `backend/app/services/medical_rules.py` | Phase 2 FastAPI routers |
| `backend/app/services/food_lookup.py` | Phase 2 FastAPI routers |
| `backend/app/services/ocr_pipeline.py` | Phase 2 FastAPI routers |
