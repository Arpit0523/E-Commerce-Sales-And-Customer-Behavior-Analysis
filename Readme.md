```markdown
# E-commerce Sales & Customer Behavior Analysis

Overview
--------
This repository implements an end-to-end e-commerce analytics pipeline:
- Synthetic data generation
- Data cleaning & master dataset creation
- Exploratory analysis and models (RFM, clustering, forecasting)
- Interactive Streamlit dashboard

Quickstart
----------
1. Create and activate venv:
   - python -m venv venv
   - source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies:
   - pip install -r requirements.txt

3. Generate sample data:
   - python generate_sample_data.py

4. Create processed master dataset:
   - python src/data_processing.py

5. Run the Streamlit dashboard:
   - streamlit run dashboard.py

Project structure
-----------------
- generate_sample_data.py  # creates data/raw/*.csv
- src/data_processing.py   # cleaning + master dataset creation
- src/analysis.py          # sales analysis utilities
- src/customer_analysis.py # customer-related analysis
- dashboard.py             # Streamlit interactive dashboard
- notebooks/               # exploratory notebooks
- data/raw/                # raw generated CSVs
- data/processed/          # processed master dataset
```