
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:

    df = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    m = df[df['Gender'] == 'Male']
    w = df[df['Gender'] == 'Female']

    cols = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    cm = {}
    cw = {}
    best = ''
    mx = -1

    for x in cols:
        a = float(m[x].corr(m['MRI_Count']))
        b = float(w[x].corr(w['MRI_Count']))
        cm[x] = a
        cw[x] = b
        if pd.notna(a) and abs(a) > mx:
            mx = abs(a)
            best = x
        if pd.notna(b) and abs(b) > mx:
            mx = abs(b)
            best = x

    return BrainCorrelationSummary(
        men_count=len(m),
        women_count=len(w),
        women_mri_correlation=cw,
        men_mri_correlation=cm,
        strongest_mri_feature=best
    )
