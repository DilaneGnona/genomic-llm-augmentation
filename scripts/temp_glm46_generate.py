import os
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IPK_X = os.path.join(BASE, '02_processed_data', 'ipk_out_raw', 'X.csv')
PEP_Y = os.path.join(BASE, '02_processed_data', 'pepper', 'y.csv')
OUTDIR = os.path.join(BASE, '04_augmentation', 'pepper', 'ipk_out_raw', 'glm46')
os.makedirs(OUTDIR, exist_ok=True)
N = 200
seed = 42

X = pd.read_csv(IPK_X)
cols = [c for c in X.columns if c != 'Sample_ID']
Xf = X.drop('Sample_ID', axis=1)
freqs = {}
for c in cols:
    vc = Xf[c].value_counts().to_dict()
    p0, p1, p2 = vc.get(0,0), vc.get(1,0), vc.get(2,0)
    n = p0 + p1 + p2
    freqs[c] = (p0/n, p1/n, p2/n) if n > 0 else (0.33, 0.34, 0.33)
rng = np.random.default_rng(seed)
arrs = []
for c in cols:
    p0,p1,p2 = freqs[c]
    arrs.append(rng.choice([0,1,2], size=N, p=[p0,p1,p2]))
syn = pd.DataFrame(np.stack(arrs, axis=1), columns=cols)
syn.insert(0, 'Sample_ID', [f'SYNTH_{i:06d}' for i in range(1, N+1)])

y = pd.read_csv(PEP_Y)[['Yield_BV']].dropna()
vals = y['Yield_BV'].values
mu_r = float(vals.mean())
sd_r = float(vals.std(ddof=1))
lo = float(vals.min())
hi = float(vals.max())
idx = rng.choice(len(vals), size=N, replace=True)
real = vals[idx]
A = 0.8 * sd_r
a = A / sd_r if sd_r > 0 else 0.0
E = sd_r**2 - A**2
E = np.sqrt(E) if E > 0 else 0.0
eps = rng.normal(0.0, E, size=N)
synY = a * (real - mu_r) + mu_r + eps
synY = np.clip(synY, lo, hi)
df_y = pd.DataFrame({'Sample_ID': syn['Sample_ID'].values, 'Yield_BV': synY})
mu_s = float(df_y['Yield_BV'].mean())
sd_s = float(df_y['Yield_BV'].std(ddof=1))
if abs(mu_s - mu_r) > 0.02:
    delta = mu_s - mu_r
    df_y['Yield_BV'] = np.clip(df_y['Yield_BV'].values - delta, lo, hi)
    mu_s = float(df_y['Yield_BV'].mean())
if abs(sd_s - sd_r) > 0.01:
    factor = (sd_r / sd_s) if sd_s > 0 else 1.0
    df_y['Yield_BV'] = np.clip(mu_s + (df_y['Yield_BV'].values - mu_s) * factor, lo, hi)
    sd_s = float(df_y['Yield_BV'].std(ddof=1))
df = syn.merge(df_y, on='Sample_ID', how='left')
out_path = os.path.join(OUTDIR, 'synthetic_glm46_ipk_200.csv')
df.to_csv(out_path, index=False)
print(out_path)
