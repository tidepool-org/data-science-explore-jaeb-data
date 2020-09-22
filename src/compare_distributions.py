import pandas as pd
from scipy import stats
import numpy as np
import utils

carb_key = "total_daily_carb_avg"  # Total daily CHO
jaeb_isf_key = "isf"
icr_key = "carb_ratio"

jaeb_path = utils.find_full_path("PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed", ".csv")
jaeb = pd.read_csv(jaeb_path)

isf = jaeb[jaeb_isf_key].replace(0, np.nan).dropna()
log_isf = np.log(isf)
# Normalize mean to 0
normed_log_isf = (log_isf - log_isf.mean(axis=0)) / log_isf.std(axis=0)
statistic, p_value = stats.kstest(normed_log_isf, 'norm')
print(statistic, p_value)

icr = jaeb[icr_key].replace(0, np.nan).dropna()
log_icr = np.log(icr)
# Normalize mean to 0
normed_log_icr = (log_icr - log_icr.mean(axis=0)) / log_icr.std(axis=0)
statistic, p_value = stats.kstest(normed_log_icr, 'norm')
print(statistic, p_value)


