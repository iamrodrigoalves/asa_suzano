import copy
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Model libraries
from models import *

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Plotly
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)

from pylab import rcParams
rcParams['figure.figsize'] = 5, 2

import warnings
warnings.filterwarnings('ignore')

import logging
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
#                                Training models                               #
# ---------------------------------------------------------------------------- #
def train_client(df, client, cut_date = '2018-11-01', tail_cut = 0):
    # Sales dataset
    volume = cut_zeros(df[client])
    volume = filter_tail(volume, t = tail_cut)
    volume = volume.resample('W').sum().resample('D').mean().fillna(0)

    vol_train = volume[volume.index < cut_date]

    # Window models
    rf = RandomForestClassifier(n_estimators=300, max_depth=3)
    try:
        mod_prob = ProbabilityModel(vol_train, client, rf)
        mod_prob.fit()
        mod_prob.optimize_windows(plot = False)
    except Exception as e:
        LOGGER.debug('\tError with prob model for %s - ' % client + str(e))
        mod_prob = None

    try:
        mod_cons = ConsumptionModel(vol_train, client)
        mod_cons.fit()
        mod_cons.optimize_windows(plot = False)
    except Exception as e:
        LOGGER.debug('\tError with cons model for %s - ' % client + str(e))
        mod_cons = None
    return (mod_prob, mod_cons)


def get_median_window(df, client):
    vol = df[client]
    vol = vol[vol > 0]
    ddays = pd.Series(vol.index).diff().dt.days
    return ddays.median()


def get_avg_window(df, client):
    vol = df[client]
    vol = vol[vol > 0]
    ddays = pd.Series(vol.index).diff().dt.days
    return ddays.mean()


def get_std_window(df, client):
    vol = df[client]
    vol = vol[vol > 0]
    ddays = pd.Series(vol.index).diff().dt.days
    return ddays.std()


def get_number_of_buys(df, client):
    vol = df[client]
    vol = vol[vol > 0]
    return len(vol)


def avg_time(datetimes):
    from datetime import datetime, date, time, timedelta
    t_s = 0
    for d in datetimes:
        t_s += (d - datetime(1970, 1, 1)).total_seconds()
        
    avg = t_s/len(datetimes)
    avg = datetime.fromtimestamp(avg)
    fday = avg.hour // 12.1
    avg += timedelta(days = int(fday))
    return avg.replace(hour = 0, minute = 0, second = 0)


def check_results_type(x):
    if isinstance(x, dict):
        return x['forecast_results']
    else:
        x.forecast_results


def merge_windows(mod_prob, mod_cons, ddays):
    mod_prob = mod_prob['forecast_results']
    mod_cons = mod_cons['forecast_results']
    if not(mod_prob is None) and not(mod_cons is None):
        results_prob = np.array([x['window'] for x in mod_prob])
        results_cons = np.array([x['window'] for x in mod_cons])
        results = [avg_time([a,b]) for a, b in zip(results_prob, results_cons)]
    elif mod_prob is None:
        results = np.array([x['window'] for x in mod_cons])
    elif mod_cons is None:
        results = np.array([x['window'] for x in mod_prob])
    return sorted(np.unique(results).tolist())


def intrapolate_windows(all_windows, ddays):
    ddays = int(2 * ddays)
    int_dates = []
    for i, r in enumerate(all_windows[:-1]):
        i_date = all_windows[i]
        f_date = all_windows[i + 1]
        int_dates += pd.date_range(start = i_date, end = f_date, freq = f'{2 * ddays}D')
    return np.unique(int_dates)


def extrapolate_windows(date, ddays, n_extra):
    ddays = int(2 * ddays)
    extra_dates = [] 
    before = pd.date_range(end = date, periods = n_extra, freq = f'{ddays}D')
    after = pd.date_range(start = date, periods = n_extra, freq = f'{ddays}D')
    extra_dates = sorted(np.unique(before.tolist() + after.tolist()))
    return extra_dates


def get_score_model(mod_prob, mod_cons, volume, n_extra, win_median, win_ratio = 0.1):
    ndays = int(round(win_median * win_ratio))
    ndays = 3 if ndays < 3 else ndays
    m_windows = merge_windows(mod_prob, mod_cons, ndays)
    windows = [] if n_extra > 0 else m_windows
    for i in m_windows:  
        windows += extrapolate_windows(i, ndays, n_extra)

    # print(f'win_ratio {win_ratio}, n_extra {n_extra}, total Windows {len(windows)}')
    
    valid_calls = 0
    total_range = []
    for i in windows:
        drange = pd.date_range(end = i, periods = ndays + 1, freq = 'D').tolist()
        drange += pd.date_range(i, periods = ndays + 1, freq = 'D').tolist()
        drange = [x for x in drange if x in volume.index]
        total_range += drange
        vol_check = (volume.loc[drange] > 0).sum() > 0
        if vol_check:
            valid_calls += 1
            
    total_range = np.unique(total_range).tolist()
    captured_vol = volume.loc[total_range]
    captured_vol = (captured_vol > 0).sum()
    return valid_calls, len(windows), captured_vol


def get_score_calls(calls, median, win_ratio = 0.1):
    ndays = int(round(median * win_ratio))
    ndays = 3 if ndays < 3 else ndays
    valid_calls = 0
    total_range = []
    for i in calls[calls.call_duration > 0].index:
        drange = pd.date_range(end = i, periods = ndays + 1, freq = 'D').tolist()
        drange += pd.date_range(i, periods = ndays + 1, freq = 'D').tolist()
        total_range += drange
        vol_check = (calls.loc[drange, 'v'] > 0).sum() > 0
        if vol_check:
            valid_calls += 1
    
    total_range = np.unique(total_range).tolist()
    captured_vol = calls.loc[total_range, 'v']
    captured_vol = (captured_vol > 0).sum()       
    return valid_calls, len(calls[calls['call_duration'] > 0]), captured_vol


def prepare_volume_df(df, client, cut_date = '2018-11-01', tail_cut = 0):
    volume = cut_zeros(df[client])
    volume = filter_tail(volume, t = tail_cut); volume.name = 'v'
    volume = volume.resample('W').sum().resample('D').mean().fillna(0)
    volume = volume[volume.index >= cut_date]
    return volume


def prepare_calls_df(df_calls, volume, client):
    # Calls dataset
    calls = df_calls[client]
    calls = pd.DataFrame(calls)
    calls.columns = ['call_duration']

    calls = pd.merge(calls.reset_index(), volume.reset_index(), on = 'OrderDate', how = 'right')
    calls = calls.set_index('OrderDate')
    return calls.sort_index()


def create_calls_plot_data(calls):
    df = calls.copy()
    df['call_flag'] = ((df['call_duration'] > 0) * 1).replace(False, np.nan)
    return go.Scatter(x = df.index, y = df.call_flag - 1, name = 'OutboundCall', mode = 'markers',
                        marker = dict(size = 10, color = (255, 255, 255)), opacity = 0.5)


def is_gap_client(df, k, cut_date = '2018-11-01', std_limit = 2):
    client_df = df[k]
    after = client_df[client_df.index >= cut_date]
    before = client_df[client_df.index < cut_date]
    
    last_buy = before[before > 0].index[-1]
    next_buy = after[after > 0].index[0]
    ddays = (next_buy - last_buy).days
    
    avg_window = get_avg_window(df[df.index < cut_date], k)
    std_window = get_std_window(df[df.index < cut_date], k)
    
    if ddays >= (avg_window + std_limit * std_window):
        return 1
    else:
        return 0