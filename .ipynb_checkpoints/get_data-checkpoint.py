# -*- coding: utf-8 -*-
"""
Created on Sun May  2 17:49:22 2021

@author: Tobia
"""

from selenium import webdriver
import time
import shutil
import os

URL = 'https://lab24.ilsole24ore.com/coronavirus/'
PATH = 'C:/Program Files (x86)/chromedriver.exe'
NEW_FRAME = '//*[@id="c19-10"]'

file_loc_d = 'C:/Users/PC/Downloads/Italia-terapie-intensive.csv'
file_loc_c = 'C:/Users/PC/Documents/Python Scripts/dati_covid/Italia-terapie-intensive.csv'

driver = webdriver.Chrome(PATH)
driver.get(URL) 

driver.execute_script("window.scrollTo(0, 12500)")
driver.switch_to.frame(driver.find_element_by_xpath(NEW_FRAME)) 
time.sleep(10)
driver.find_element_by_xpath('//*[@id="scarica-dati"]').click()
time.sleep(3)
driver.close()

if os.path.isfile(file_loc_c):
    os.remove(file_loc_c)
    
time.sleep(2)
shutil.move(file_loc_d, file_loc_c)
