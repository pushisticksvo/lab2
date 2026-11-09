
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:

    df = pd.read_csv(data.csv_path)

    age = {}
    surv = {}
    for k, t in df.groupby('Pclass'):
        age[int(k)] = float(t['Age'].mean())
        surv[int(k)] = float(t['Survived'].mean())

    return TitanicSummary(
        row_count=len(df),
        missing_by_column=df.isna().sum().astype(int).to_dict(),
        adults_over_30_count=int((df['Age'] > 30).sum()),
        mean_age_by_pclass=age,
        survival_rate_by_pclass=surv,
        highest_fares=df['Fare'].nlargest(5).astype(float).tolist()
    )
