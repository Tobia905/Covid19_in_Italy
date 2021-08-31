import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime

def diff_dates(x, y, dateuse = 'pandas', period = None):
    
    diff = y - x
    if period is None:
        return diff
    
    else:
        time = []
        approx_month = (1/12)*28 + (4/12)*30 + (7/12)*31
        period_dict = {'days':1, 'months':approx_month, 'years':365}
        for d in diff:
            if dateuse == 'pandas':
                time.append(int(d.days / period_dict[period]))
            else:
                time.append(d / np.timedelta64(1, 'D'))
                time = np.array(time) / period_dict[period]
                time = np.around(time, 0).tolist()
                
    return time

def get_weekday(date):

    day_map = {
        0:'Monday',
        1:'Tuesday',
        2:'Wednesday',
        3:'Thursday',
        4:'Friday',
        5:'Saturday',
        6:'Sunday'
    }

    date = pd.to_datetime(date)
    date = [d.weekday() for d in date]

    return pd.Series(date).map(day_map)

def dt2cal(dt):
    
    # allocate output 
    out = np.empty(dt.shape + (7,), dtype="u4")
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    
    # decompose calendar floors
    out[..., 2] = (D - M) + 1 # day
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    
    return out

def get_equal_date(datelist):
    
    datelist = list(datelist.astype('str'))
    
    def converter(datelist):
        for i in datelist:
            try:
                yield parser.parse(i)
            except ValueError:
                try:
                    yield parser.parse(i, dayfirst=True)
                except ValueError:
                    try:
                        yield datetime.strptime(i, '%Y-%d-%b')
                    except:
                        yield i

    dat = list(converter(list(str(datelist))))
    
    return [i.strftime('%d-%m-%Y') for i in dat]
