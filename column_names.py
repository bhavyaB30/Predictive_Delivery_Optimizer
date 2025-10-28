import pandas as pd
import os

data_folder = "data"


csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

print("🔍 Checking column names for all CSV files...\n")

for file in csv_files:
    try:
        path = os.path.join(data_folder, file)
        df = pd.read_csv(path)
        print(f"📁 {file}")
        print("➡️ Columns:", df.columns.tolist(), "\n")
    except Exception as e:
        print(f"⚠️ Could not read {file}: {e}\n")
