import numpy as np
import pandas as pd
from dateutil import parser
from datetime import datetime

def diff_dates(x,y, dateuse = 'pandas', days = False, months = False, years = False):
    '''
    Takes two array-like obj. of datetimes and 
    gives the differences in days, months
    or years between dates.
    ------------------
    input: 
    x,y (arrays of datetimes)
    days (boolean, default = False)
    months (boolean, default = False)
    years (boolean, deafault = False)
    output:
    array of floats.
    '''
    diff = y - x
    if dateuse == 'pandas':
        
        if days:
            diff_days = []
            for day in diff:
                diff_days.append(day.days)
            return diff_days
        
        elif months:
            diff_months = []
            for month in diff:
                diff_months.append(int(month.days / ((1/12)*28 + (4/12)*30 + (7/12)*31)))
            return diff_months
        
        elif years:
            diff_years = []
            for year in diff:
                diff_years.append(year.days / 365)
            return diff_years
        
        else:
            return diff
        
    elif dateuse == 'numpy':
        if days: 
            day = []
            for d in diff:
                day.append(d / np.timedelta64(1, 'D'))
            return np.array(day)
        
        elif months:
            month = []
            for d in diff:
                month.append(d / np.timedelta64(1, 'D'))
            # here i divide by a weighted average of number of days in months
            m = np.array(month) / ((1/12)*28 + (4/12)*30 + (7/12)*31)
            return np.around(m, 0)
        
        elif years:
            year = []
            for d in diff:
                year.append(d / np.timedelta64(1, 'D'))
            y = np.array(year) / 365
            return np.around(y, 0)
        
        else:
            return diff
        
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
