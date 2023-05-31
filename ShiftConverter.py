import os
import random
from time import time
import numpy as np
import gc
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self,
                 paths,
                 shuffle=True,
                 to_memory=False,
                 save=False,
                 savedir='train'):
        self.shuffle = shuffle
        self.to_memory = to_memory
        self.save = save
        self.savedir = savedir
        self.data = {}

        # загрузка всего датасета в память
        for path in paths:
            for root, _, files in os.walk(path):
                for name in files:
                    filename = name.split('.')
                    key = (filename[0] + filename[1][:2], filename[4]) # ключ – кортеж вида (дата и час, луч)
                    if to_memory:
                        self.data[key] = np.load(os.path.join(root, name))
                    else:
                        self.data[key] = os.path.join(root, name)

    def __call__(self):
        target_datetime = list(self.data.keys())
        if self.shuffle:
            random.shuffle(target_datetime)

        # итерация по ключам в словаре self.data, где ключи – название файла
        for key in target_datetime:
            for shift in range(60):
                x, y = self.__getArray(key, shift)
                if x is not None and y is not None:
                    # print(f'key is {key} shift is {shift}')

                    # сохранение подготовленных данных
                    if self.save:
                        np.save(
                            os.path.join(self.savedir, f'x-{key[0]}-beam{key[1]}-shift{shift}.npy'),
                            np.concatenate([x[:,:,2:3], x[:,:,1:2]], axis=-1))
                        np.save(
                            os.path.join(self.savedir, f'y-{key[0]}-beam{key[1]}-shift{shift}.npy'),
                            y[:,:,2:3]*y[:,:,1:2])
                        yield (key, shift)
                    else:
                        yield np.concatenate([x[:,:,2:3], x[:,:,1:2]], axis=-1), y[:,:,2:3]*y[:,:,1:2]
                else:
                    del x
                    del y
                    gc.collect()
                    continue

    def __getArray(self, key, shift=0):
        if shift >= 60:
            # print('shift argument must be less than regular records window resolution (60)!')
            return None

        y = None

        if shift > 0:
            shifted_datetime = datetime.strptime(key[0], '%Y%m%d%H')
            shifted_key = ((shifted_datetime+timedelta(hours=2)).strftime('%Y%m%d%H'), key[1])
            if tuple(key) in self.data and tuple(shifted_key) in self.data:
                arr, shifted_arr = None, None
                if self.to_memory:
                    arr = self.data[key][:, shift:]
                    shifted_arr = self.data[shifted_key][:, :shift]
                else:
                    arr = np.load(self.data[key])[:, shift:]
                    shifted_arr = np.load(self.data[shifted_key])[:, :shift]
                y = np.concatenate([arr, shifted_arr], axis=1)
            else:
                gc.collect()
                return [None, None]
        else:
            if self.to_memory:
                y = self.data[key]
            else:
                y = np.load(self.data[key])

        day = self.__getDayArray(key, shift)
        week = self.__getWeekArray(key, shift)

        if day is not None and week is not None:
            return [np.concatenate([day, week], axis=1), y]
        else:
            return [None, None]

    def __getDayArray(self, key, shift=0):
        if shift >= 60:
            print('shift argument must be less than regular records window resolution (60)!')
            return None

        seq = self.__getDayNames(key)
        arrays = []
        for item in seq:
            if tuple(item) in self.data:
                if self.to_memory:
                    arrays.append(self.data[item])
                else:
                    arrays.append(np.load(self.data[item]))
            else:
                del arrays
                gc.collect()
                return None
        
        if shift > 0:
            if self.to_memory:
                arrays.append(self.data[key])
            else:
                arrays.append(np.load(self.data[key]))
            arrays[0] = arrays[0][:, shift:]
            arrays[-1] = arrays[-1][:, :shift]
            return np.concatenate(arrays, axis=1)
        else:
            return np.concatenate(arrays, axis=1)
        
    def __getWeekArray(self, key, shift=0):
        if shift >= 60:
            # print('shift argument must be less than regular records window resolution (60)!')
            return None
        
        seq = self.__getWeekNames(key)
        arrays = []
        if shift > 0:
            shifted_datetime = datetime.strptime(key[0], '%Y%m%d%H')
            shifted_key = ((shifted_datetime+timedelta(hours=2)).strftime('%Y%m%d%H'), key[1])
            shifted_seq = self.__getWeekNames(shifted_key)

            for item, shifted_item in zip(seq, shifted_seq):
                if tuple(item) in self.data and tuple(shifted_item) in self.data:
                    arr, shifted_arr = None, None
                    if self.to_memory:
                        arr = self.data[item][:, shift:]
                        shifted_arr = self.data[shifted_item][:, :shift]
                    else:
                        arr = np.load(self.data[item])[:, shift:]
                        shifted_arr = np.load(self.data[shifted_item])[:, :shift]
                    arrays.append(np.concatenate([arr, shifted_arr], axis=1))

                else:
                    del arrays
                    gc.collect()
                    return None
                 
        else:
            for item in seq:
                if tuple(item) in self.data:
                    if self.to_memory:
                        arrays.append(self.data[item])
                    else:
                        arrays.append(np.load(self.data[item]))
                else:
                    del arrays
                    gc.collect()
                    return None

        return np.concatenate(arrays, axis=1)

    def __getDayNames(self, key):
        filename_datetime = datetime.strptime(key[0], '%Y%m%d%H')

        # список массивов за день до целевого массива
        dayBefore = []
        for i in range(24, 0, -2):
            hoursBefore = ((filename_datetime-timedelta(hours=i)).strftime('%Y%m%d%H'), key[1])
            dayBefore.append(hoursBefore)
        
        return dayBefore
    
    def __getWeekNames(self, key):
        filename_datetime = datetime.strptime(key[0], '%Y%m%d%H')

        # тот же час, но за неделю до целевого массива
        weekBeforeInThatHour = []
        for i in range(7, 1, -1):
            thatHour = ((filename_datetime-timedelta(days=i)).strftime('%Y%m%d%H'), key[1])
            weekBeforeInThatHour.append(thatHour)

        return weekBeforeInThatHour
    
if __name__ == '__main__':
    train_loader = DataLoader(paths=['2018-converted/2018-01',
                                     '2018-converted/2018-02',
                                     '2018-converted/2018-03',
                                     '2018-converted/2018-04',
                                     '2018-converted/2018-05',
                                     '2018-converted/2018-06',
                                     '2018-converted/2018-07',],
                              save=True,
                              savedir='train',)
    val_loader = DataLoader(paths=['2018-converted/2018-08',
                                   '2018-converted/2018-09',],
                              save=True,
                              savedir='validation',)
    
    train_keys = list(train_loader.__call__())
    val_keys = list(val_loader.__call__())

    print(f'Train size is {len(train_keys)}')
    print(f'Validation size is {len(val_keys)}')