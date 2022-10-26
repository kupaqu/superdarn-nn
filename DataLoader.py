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
                 datapath: str,
                 windows_num: int = 30,
                 shuffle = True):

        self.datapath = os.path.abspath(datapath)    # путь к данным
        self.windows_num = windows_num  # количество периодов в окне
        self.shuffle = shuffle
        
        self.sequence_for_learning = self.__get_sequence_for_learning()
        
        for filename in self.sequence_for_learning[0]:
            if filename.endswith('patch'):
                continue

            x = np.load(filename)
            self.shape = x.shape
            break

    def __call__(self):
        if self.shuffle:
            random.shuffle(self.sequence_for_learning)
        
        for sequence in self.sequence_for_learning:
            x, y = self.__xy(sequence)
            for beam in range(16):
                yield x[:,:,:,beam,0], y[:,:,:,beam,0]

    def __xy(self, sequence):
        if len(sequence) < 2:
            raise('Learning sequence must be at least 2 elements!')
        
        timeseries = []

        for filename in sequence:
            if filename.endswith('patch'):
                timeseries.append(np.zeros(shape=self.shape))
            else:
                timeseries.append(np.load(filename))

        x = np.concatenate(timeseries[:-1], axis=1)
        y = timeseries[-1]
        
        return x, y

    def __get_sequence_for_learning(self, augmentation_step=1):

    #
    #   Генерирует последовательности имен файлов для обучения,
    #   подпоследовательности упорядочены, что позволяет
    #   в последующем шаффлить датасет
    #
        sequence = []

        # подпапки с радарами
        radars = next(os.walk(self.datapath))[1]
        
        # итерация по папкам с радарами
        for radar in radars:
            radar_subdir = os.path.join(self.datapath, radar)

            # итерация по периодам
            for hour in range(0, 24, 2):
                
                # вытаскиваем все имена файлов за период указаный в переменной hour
                filenames = self.__get_filenames(radar_subdir, hour)

                # итерация скользящим окном по файлам
                for i in range(0, len(filenames)-self.windows_num, augmentation_step):

                    # целевое значение не должно быть заплаткой
                    if filenames[i+self.windows_num].endswith('patch'):
                        continue
                    
                    # хотя бы половина контекстных данных должны присутствовать
                    sub_sequence = filenames[i:i+self.windows_num+1] # последний элемент - целевое значение
                    if self.__count_patches(sub_sequence) > len(sub_sequence) / 4:
                        continue
                    
                    sequence.append(sub_sequence)

        return sequence
                    
    def __count_patches(self, sequence):
        cnt = 0

        for item in sequence:
            if item.endswith('.patch'):
                cnt += 1
        
        return cnt

    def __get_filenames(self,
                        directory: str, # папка, где лежат файлы
                        hour: int):     # время суток

        #
        #   Возвращает упорядоченный список всех имен файлов,
        #   относящихся к одному и тому же времени суток.
        #   Пропущенные даты заменяются строкой-"заглушкой".
        #

        formatted_time = datetime.time(hour=hour, minute=0)
        filenames = sorted(glob.glob(directory+f'/*/*/*.{formatted_time.strftime("%H")}*.00.*.npy', recursive=True))
        # массив для пропущенных значений
        patches = []
        # проверка остутствия пропусков во временном ряде и генерация заглушек
        if filenames:

            sliding_date = self.__get_date_from_filename(filenames[0])
            delta = datetime.timedelta(days=1)

            for i in range(len(filenames)):

                while self.__get_date_from_filename(filenames[i]) != sliding_date:
                    warnings.warn(f'There is no data for {hour} hours at {sliding_date} in {directory}!')
                    patches.append(f"{directory}/{sliding_date.strftime('%Y/%Y-%m/%Y%m%d')}.patch")
                    sliding_date += delta

                sliding_date += delta
            
            # cоединяем список файлов с заплатками
            filenames += patches

        return sorted(filenames)

    def __get_date_from_filename(self, filename):

        #
        #   Возвращает дату указанную в файле.
        #

        return datetime.datetime.strptime(filename.split('/')[-1].split('.')[0], '%Y%m%d')