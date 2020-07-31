import pandas as pd
import utils
from pathlib import Path

base_path = Path(__file__).parent
data_path = (
    base_path
    / "../data/PHI-issue-reports-with-surrounding-2week-data-summary-stats-2020-07-28.csv"
).resolve()
df = pd.read_csv(data_path)

df = df[(df.percent_70_180 >= 80)]

print(df.count)
export_path = (base_path / "../results/PHI-filtered-subjects-80-100.csv").resolve()
df.to_csv(export_path)
