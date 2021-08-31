import math
from collections import Counter
import numpy as np
import pandas as pd
import scipy.stats as ss

def cramers_v(x, y):
    
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
        
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def get_sign_matrix(x, y = '', z = '', sparse = False):

    v = x[y].unique()
    rep = []
    for t in v: # t in group that you are testing 
        rep.append(x[x[y] == t][z])
        
    test_val = []
    test_p = [] 
    for group in rep:
        #value of the test
        test_list_v = []
        #p_values of the test
        test_list_p = []
        for i in range(len(rep)): 
            # create a list of ranksums test's values
            test_list_v.append(ss.ranksums(rep[i],group)[0])
            # create a list of ranksums test's p-values
            test_list_p.append(ss.ranksums(rep[i],group)[1])
            
        # this creates a list of lists, containing all the pairwise comparison
        test_val.append(test_list_v)
        test_p.append(test_list_p)
        
    test_df_p = pd.DataFrame(test_p, index = v, columns = v)
    test_df_v = pd.DataFrame(test_val, index = v, columns = v)
    
    if sparse:
    
        # lambda that assign 1 when p-value is over 0.05
        sign = lambda x: 0 if x <=0.05 else 1
        
        for col in test_df_p.columns:
            test_df_p[col] = test_df_p[col].apply(sign)
            
    return test_df_v, test_df_p
        
def great_circle_dist(coord1, coord2, unit = 'km'):
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * \
                                                           math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    if unit == 'km':
        return round(km, 3)
    
    elif unit == 'meters':
        return round(meters)
    

    
    
    
    
    
    
    
    
    
    
    
    