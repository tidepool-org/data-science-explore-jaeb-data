import pandas as pd
from scipy import stats
import utils

carb_key = "total_daily_carb_avg"  # Total daily CHO
t1d_isf_key = "InsulinSensitivity"
jaeb_isf_key = "isf"

t1d_exchange_path = utils.find_full_path("t1d_exchange", ".csv")
t1d_exchange = pd.read_csv(t1d_exchange_path)

jaeb_path = utils.find_full_path("PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed", ".csv")
jaeb = pd.read_csv(jaeb_path)

print("Carbs:\n", stats.ks_2samp(t1d_exchange[carb_key], jaeb[carb_key]))
print("ISF:\n", stats.ks_2samp(t1d_exchange[t1d_isf_key], jaeb[jaeb_isf_key]))


