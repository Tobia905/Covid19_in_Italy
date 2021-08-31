# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:59:58 2021

@author: Tobia
"""

import pandas as pd
from ICD9_retrieving import DIA, INT, DEATH

CONVERTERS = {'diagnosis': DIA, 'intervention': INT, 'death': DEATH}

def ICD_converter(x, code = 'diagnosis'):
    conv = CONVERTERS[code]
    if not isinstance(x, pd.core.series.Series):
        x = pd.Series(x)
    return x.map(conv)


