import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

df = pd.read_csv(r"""dataset.csv""")
# Attempt to parse date column automatically
for c in df.columns:
    try:
        parsed = pd.to_datetime(df[c], errors='coerce')
        if parsed.notna().sum() > len(df)*0.3:
            df[c] = parsed
            df = df.set_index(c).sort_index()
            break
    except Exception:
        pass

numeric = df.select_dtypes(include=[np.number]).columns.tolist()
os.makedirs("output", exist_ok=True)

for col in numeric:
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df[col])
    plt.title(f"Time Series: {col}" )
    plt.xlabel("Date" if hasattr(df.index, 'dtype') else "Index")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join("output", f"time_series_{col.replace(' ','_')}.png"))
    plt.close()

# Rolling stats for first numeric
if numeric:
    col = numeric[0]
    s = df[col].dropna().astype(float)
    rm = s.rolling(window=12, min_periods=1).mean()
    rs = s.rolling(window=12, min_periods=1).std()
    plt.figure(figsize=(10,4))
    plt.plot(s.index, s, label=col)
    plt.plot(rm.index, rm, label='12-period rolling mean')
    plt.plot(rs.index, rs, label='12-period rolling std')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join("output", f"rolling_stats_{col.replace(' ','_')}.png"))
    plt.close()

# Save basic stats
df[numeric].describe().transpose().to_csv("basic_stats.csv")
print("Output saved to ./output and basic_stats.csv")
