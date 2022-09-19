import os
import requests
from datetime import datetime, timedelta, time
from tqdm import tqdm

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

URL = "https://chapman.usask.ca/data/"
RADAR = "sas"
START_TIME = datetime(2022, 7, 1, 0, 0, 0)
END_TIME = datetime(2022, 9, 1, 0, 0, 0)
EXEC_DIR = os.path.dirname(__file__)
REL_DIR = "data"

print("Downloading progress:")
for single_date in tqdm(daterange(START_TIME, END_TIME)):
    year, month, day = single_date.strftime("%Y %m %d").split()
    for h in range(0, 24, 2):
        m = 0
        # for m in range(0, 5):
        hm = time(h, m).strftime("%H%M")
        filename = f'{year}{month}{day}.{hm}.00.{RADAR}.a.fitacf.bz2'
        full_url = f'{URL}/{year}/{month}/{filename}'
        r = requests.get(full_url, allow_redirects=True)
        if r.status_code == 200:
            f = open(EXEC_DIR+'/'+REL_DIR+'/'+filename, 'wb')
            f.write(r.content)
            f.close()
        # else:
        #     print('\n' + full_url + ' NOT FOUND')