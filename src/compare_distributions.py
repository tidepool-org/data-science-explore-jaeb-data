import pandas as pd
from scipy import stats
import numpy as np
import utils

carb_key = "total_daily_carb_avg"  # Total daily CHO
t1d_isf_key = "InsulinSensitivity"
jaeb_isf_key = "isf"

t1d_exchange_path = utils.find_full_path("t1d_exchange", ".csv")
t1d_exchange = pd.read_csv(t1d_exchange_path)

jaeb_path = utils.find_full_path("PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed", ".csv")
jaeb = pd.read_csv(jaeb_path)

isf = jaeb[jaeb_isf_key].replace(0, np.nan).dropna()
log_isf = np.log(isf)
statistic, p_value = stats.kstest(log_isf, 'norm')
print(statistic, p_value)

k, p_2 = stats.normaltest(log_isf)
print(k, p_2)


