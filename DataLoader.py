import os
import random
from time import time
import numpy as np
import datetime
import glob
import warnings
from tqdm import tqdm

class DataLoader:
    def __init__(self,
                 paths,
                 shuffle=True):
        self.shuffle = shuffle
        self.data = {}

        # загрузка всего датасета в память
        for path in paths:
            for root, _, files in os.walk(path):
                for name in files:
                    filename = name.split('.')
                    key = (filename[0] + filename[1][:2], filename[4]) # ключ – кортеж вида (дата и час, луч)
                    arr = np.load(os.path.join(root, name))
                    self.data[key] = arr

    def __call__(self):
        target_datetime = list(self.data.keys())
        if self.shuffle:
            random.shuffle(target_datetime)

        # итерация по ключам в словаре self.data, где ключи – название файла
        for key in target_datetime:
            seq = self.__getSequence(key) # ключи исторических данных
            arrays = []
            missData = False

            for item in seq:
                try:
                    arrays.append(self.data[item])

                # некоторые исторические данные могут отсутствовать
                except KeyError:
                    missData = True
                    break
            
            # если есть пропуски, то пропускаем пример
            if missData:
                continue
            else:
                x = np.concatenate(arrays, axis=1)
                y = self.data[key]
                yield np.concatenate([x[:,:,2:3], x[:,:,1:2]], axis=-1), y[:,:,2:3]*y[:,:,1:2] 

    def __getSequence(self, key):
        filename_datetime = datetime.strptime(key[0], '%Y%m%d%H')

        # список массивов за день до целевого массива
        dayBefore = []
        for i in range(24, 0, -2):
            hoursBefore = ((filename_datetime-timedelta(hours=i)).strftime('%Y%m%d%H'), key[1])
            dayBefore.append(hoursBefore)

        # тот же час, но за неделю до целевого массива
        weekBeforeInThatHour = []
        for i in range(7, 1, -1):
            thatHour = ((filename_datetime-timedelta(days=i)).strftime('%Y%m%d%H'), key[1])
            weekBeforeInThatHour.append(thatHour)

        return dayBefore + weekBeforeInThatHour

        # список массивов за неделю до целевого массива
        # weekBefore = []
        # for i in range(24*7, 0, -2):
        #     hoursBefore = ((filename_datetime-timedelta(hours=i)).strftime('%Y%m%d%H'), key[1])
        #     weekBefore.append(hoursBefore)
        
        return weekBefore
