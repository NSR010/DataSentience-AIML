import pandas as pd, matplotlib.pyplot as plt, numpy as np
df = pd.read_csv(r"""dataset.csv""")
for c in df.columns:
    try:
        parsed = pd.to_datetime(df[c], errors='coerce')
        if parsed.notna().sum() > len(df)*0.3:
            df[c] = parsed; df = df.set_index(c).sort_index(); break
    except Exception:
        pass
print(df.head())
print(df.describe())
