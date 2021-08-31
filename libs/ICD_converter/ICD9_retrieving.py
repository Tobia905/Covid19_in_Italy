# coding: utf-8



import pandas as pd
import re

icd9_1 = pd.read_excel('./icd_codes/CMS32_DESC_LONG_SHORT_DX.xlsx')
icd9_2 = pd.read_excel('./icd_codes/CMS32_DESC_LONG_SHORT_SG.xlsx', converters={'PROCEDURE CODE':str})
icd10 = pd.read_csv('./icd_codes/icd10cm_codes_2021.txt',
                    sep = '\t', header = None).rename({0: 'Description'}, axis = 1)
DIA = pd.Series(icd9_1['LONG DESCRIPTION'].values, index = icd9_1['DIAGNOSIS CODE']).to_dict()
INT = pd.Series(icd9_2['LONG DESCRIPTION'].values, index = icd9_2['PROCEDURE CODE']).to_dict()

icd10['Code'] = list(map(lambda x: re.findall(r'^\D\d{1,10}[\s,a-zA-Z]\D|^\D\d{1,10}\D\d{1,10}', x),
                         icd10.Description))
icd10['Code'] = pd.Series([x[0] for x in icd10.Code]).replace('\s{1,5}', '', regex = True)
icd10.Description = pd.Series(list(map(lambda x: re.sub(r'^\D\d{1,10}[\s,a-zA-Z]\D|^\D\d{1,10}\D\d{1,10}', '', x),
                                       icd10.Description))).replace('^\s{1,5}|\s\d', '', regex = True)

DEATH = pd.Series(icd10.Description.values, index = icd10.Code).to_dict()











