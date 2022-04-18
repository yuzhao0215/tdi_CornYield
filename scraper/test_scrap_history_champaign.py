import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import time

pd.options.display.max_columns = None
pd.options.display.max_rows = None
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service


# Use .format(YYYY, M, D)
lookup_URL = 'https://www.wunderground.com/history/daily/us/{}/{}/date/{}-{}-{}'
start_date = date(1978, 1, 1)
end_date = start_date + pd.Timedelta(days=3)

state = 'il'
county = 'urbana'

df_prep = pd.DataFrame()

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')

driver = webdriver.Chrome(service=Service('./chromedriver.exe'), options=options)

while start_date != end_date:
    print('gathering data from: ', start_date)

    formatted_lookup_URL = lookup_URL.format(state,
                                             county,
                                             start_date.year,
                                             start_date.month,
                                             start_date.day)

    driver.get(formatted_lookup_URL)

    table_element = WebDriverWait(driver, 60).\
        until(
        EC.
            visibility_of_all_elements_located((By.XPATH, '//tr[@class="ng-star-inserted"]')))

    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'lxml')

    tbody_elements = soup.findAll('tbody', class_='ng-star-inserted')
    print('lengh of tbodys: {}'.format(len(tbody_elements)))

    names = ['date']
    values = [start_date.strftime("%m/%d/%Y")]

    for i, el in enumerate(tbody_elements):
        row_elements = el.findAll('tr')

        for rw in row_elements:
            attri_name = rw.find('th').text

            if re.match(r".*(twilight.*)|(.*moon.*)", attri_name, re.IGNORECASE):
                continue

            attri_val = rw.find('td', class_='ng-star-inserted').text

            names.append(attri_name)
            values.append(attri_val)

    df_prep = pd.concat([
        df_prep,
        pd.DataFrame([values], columns=names)
    ], ignore_index=True)
 
    start_date += timedelta(days=1)
    # print(formatted_lookup_URL)
    time.sleep(10)

print(df_prep)
