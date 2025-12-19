import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

def run_automation():
    """
    Mengambil data dari UCI dan melakukan preprocessing sesuai notebook eksperimen.
    Target: Kriteria 1 (Skilled/Advance)
    """
    print("Mengambil dataset dari UCI Repo (ID: 529)...")
    try:
        diabetes = fetch_ucirepo(id=529)
        X = diabetes.data.features
        y = diabetes.data.targets
        
        print("Melakukan Encoding (get_dummies)...")
        X_encoded = pd.get_dummies(X, drop_first=True)
        y_encoded = y['class'].map({'Positive': 1, 'Negative': 0})
        
        df_clean = pd.concat([X_encoded, y_encoded], axis=1)
        
        output_dir = "preprocessing/diabetes_preprocessing"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "diabetes_cleaned.csv")
        df_clean.to_csv(output_file, index=False)
        
        print(f"Otomatisasi Berhasil! Data disimpan di: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_automation()