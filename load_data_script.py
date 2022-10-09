from datetime import datetime, timedelta, time
import requests
import os

os.makedirs('val_data')
os.makedirs('data')

def daterange(start_date, end_date):
  for n in range(int((end_date - start_date).days)):
    yield start_date + timedelta(n)

# данные для обучения
start_date = datetime(2022, 1, 1, 0, 0, 0)
end_date = datetime(2022, 5, 1, 0, 0, 0)
for single_date in daterange(start_date, end_date):
  year, month, day = single_date.strftime("%Y %m %d").split()
  for h in range(0, 24, 2):
      m=0
    # for m in range(0, 5):
      hm = time(h, m).strftime("%H%M")
      url = f'https://chapman.usask.ca/data/{year}/{month}/{year}{month}{day}.{hm}.00.inv.a.fitacf.bz2'
      r = requests.get(url, allow_redirects=True)
      if r.status_code == 200:
        filename = f'data/{year}{month}{day}.{hm}.00.inv.a.fitacf.bz2'
        print(filename)
        f = open(filename, 'wb')
        f.write(r.content)
        f.close()

# данные для валидации
# данные для обучения
start_date = datetime(2022, 1, 1, 0, 0, 0)
end_date = datetime(2022, 5, 1, 0, 0, 0)
for single_date in daterange(start_date, end_date):
  year, month, day = single_date.strftime("%Y %m %d").split()
  for h in range(0, 24, 2):
      m=0
    # for m in range(0, 5):
      hm = time(h, m).strftime("%H%M")
      url = f'https://chapman.usask.ca/data/{year}/{month}/{year}{month}{day}.{hm}.00.inv.a.fitacf.bz2'
      r = requests.get(url, allow_redirects=True)
      if r.status_code == 200:
        filename = f'val_data/{year}{month}{day}.{hm}.00.inv.a.fitacf.bz2'
        print(filename)
        f = open(filename, 'wb')
        f.write(r.content)
        f.close()