# Liver-Cirrhosis-Prediction-using-Random-Forest-
Liver Cirrhosis Prediction using Random Forest applies machine learning to analyze clinical data like age, bilirubin, enzymes, and proteins to predict cirrhosis risk. The Random Forest algorithm combines multiple decision trees for high accuracy, enabling early detection, timely treatment, and improved patient outcomes.
Liver Cirrhosis Prediction — Streamlit + Random Forest
Modules:

Home (Project overview & synopsis with image)
Prediction (Yes/No)
Visualization (13 common plots)
This app is built with Streamlit and RandomForestClassifier. It uses a Kaggle-compatible liver dataset format (similar to ILPD). No database is used.

How to run
pip install -r requirements.txt
streamlit run app.py
The app starts in your browser.

Dataset
You can upload a Kaggle liver dataset (e.g., ILPD CSV).
Or, tick "Use sample dataset" to load the provided synthetic sample with 600 rows:
data/sample_liver_ilpd_like_600.csv
Expected columns (ILPD-like):

Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
Alanine_Aminotransferase, Aspartate_Aminotransferase, Total_Proteins,
Albumin, Albumin_and_Globulin_Ratio, Dataset
Target column: Dataset where 1 = patient (Yes) and 2 = not patient (No).
Notes
The Random Forest is trained on the loaded dataset each run for simplicity.
All processing is in-memory; no database is used.
Visualization module provides 13 plot types.
File Tree
liver_cirrhosis_rf_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── home_banner.png
└── data/
    └── sample_liver_ilpd_like_600.csv
