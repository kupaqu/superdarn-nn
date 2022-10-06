import os
import numpy as np
import datetime as dt
import glob
from warnings import warn
import bz2
import pydarn

class DataLoader:

    def __init__(self,
                 datapath: str,
                 windows_num: int = 30):

        self.datapath = datapath        # путь к данным в формате fitacf.bz2
        self.windows_num = windows_num  # количество периодов в окне
        self.delta_hours = 2            # разница в часах между началом записи в файле 
        self.nrang = 75                 # количество гейтов
        self.keys = ['pwr0',            # ключи данных
                     'v',
                     'p_l',
                     'p_s',
                     'w_l',
                     'w_s',
                     'qflg']       # ключ по которому лежит маска
        self.total_bms = 16             # количество лучей
        self.total_chs = 2              # количество каналов
        self.reg_res = 60               # регулярное разрешение (количество записей за два часа)

    def __iter__(self):

    #
    #   Функция вызывающаяся при каждой итерации.
    #   Возвращает окно заданного размера,
    #   содержащее сконкатенированные окна
    #   за определенное время суток.
    #

        # итерация по подпапкам данных каждого из радаров
        for radar_subdir, _, _ in os.walk(self.datapath):

            # итерация по периодам
            for hour in range(0, 24, self.delta_hours):

                # вытаскиваем все имена файлов за период указаный в переменной hour
                filenames = self.__get_filenames(radar_subdir, hour)
                # итерация скользящим окном по файлам
                for i in range(len(filenames)-self.windows_num):

                    # целевое значение не должно быть заплаткой
                    if filenames[i+self.windows_num].endswith('patch'):
                        continue
                    
                    # итерация по номерам каналов
                    for channel in range(self.total_chs):

                        # итерация по номерам лучей
                        for beam in range(self.total_bms):
                            
                            # извлечение данных в виде многомерного массива
                            x = self.__get_window(filenames=filenames[i:(i+self.windows_num)],
                                                                 bmnum=beam,
                                                                 channel=channel)

                            # извдечение целевого значения
                            y = self.__get_data(filenames[i+self.windows_num],
                                                          bmnum=beam,
                                                          channel=channel)

                            yield (x, y)
    
    def __get_filenames(self,
                        directory: str, # папка, где лежат файлы
                        hour: int):     # время суток

        #
        #   Возвращает упорядоченный список всех имен файлов,
        #   относящихся к одному и тому же времени суток.
        #   Пропущенные даты заменяются строкой-"заглушкой".
        #

        formatted_time = dt.time(hour=hour, minute=0)
        filenames = sorted(glob.glob(directory+f'/*.{formatted_time.strftime("%H%M")}.00.*.fitacf.bz2'))
        # массив для пропущенных значений
        patches = []

        # проверка остутствия пропусков во временном ряде и генерация заглушек
        if filenames:

            sliding_date = self.__get_date_from_filename(filenames[0])
            delta = dt.timedelta(days=1)

            for i in range(len(filenames)):

                while self.__get_date_from_filename(filenames[i]) != sliding_date:
                    warn(f'There is no data for {hour} hours 0 minutes at {sliding_date} in {directory}!')
                    patches.append(f"{directory}/{sliding_date.strftime('%Y%m%d')}.patch")
                    sliding_date += delta

                sliding_date += delta
            
            # cоединяем список файлов с заплатками
            filenames += patches

        return sorted(filenames)

    def __get_date_from_filename(self, filename):

        #
        #   Возвращает дату указанную в файле.
        #

        return dt.datetime.strptime(filename.split('/')[-1].split('.')[0], '%Y%m%d')

    def __get_window(self,
                     filenames: list,
                     bmnum: int,
                     channel: int):

        #
        #   Функция конкатенирующая выводы функции __get_data
        #   в большие, общие массивы.
        #

        timeseries = np.zeros(shape=(self.nrang, 0, len(self.keys)))

        for filename in filenames:
            sub_series = self.__get_data(filename, bmnum, channel)
            timeseries = np.concatenate((timeseries, sub_series), axis=1)

        return timeseries
    
    def __get_data(self,
                   filename: str,   # имя файла из которого извлекаются данные
                   bmnum: int,      # номер луча
                   channel: int):   # номер канала
        
        #
        #   Извлекает из файла массив данных.
        #

        print(filename)

        timeseries = np.zeros(shape=(self.nrang, self.reg_res, len(self.keys)))

        # если вместо записи присутствует заплатка, возвращаем пустые массивы
        if filename.endswith('patch'):
            return timeseries

        # открытие файла на чтение
        with bz2.open(filename) as fp:
            fitacf_stream = fp.read()
        reader = pydarn.SuperDARNRead(fitacf_stream, True)
        records = reader.read_fitacf()

        # список меток времени
        timestamps = []
        # счетчик, показывающий текущий индекс в массивах timeseries и mask
        cur = 0

        for i in range(len(records)):

            # если номер луча и канала совпадают с нужным
            if records[i]['bmnum'] == bmnum and records[i]['channel'] == channel:
                
                rec_time = dt.datetime(year=records[i]['time.yr'],
                        month=records[i]['time.mo'],
                        day=records[i]['time.dy'],
                        hour=records[i]['time.hr'],
                        minute=records[i]['time.mt'])

                if timestamps != []:

                    diff_mins = (rec_time - timestamps[-1]).seconds // 60

                    if diff_mins == 2:
                        timestamps.append(rec_time)
                        if 'slist' in records[i]:
                            for n, m in enumerate(records[i]['slist']):
                                for j, k in enumerate(self.keys):
                                    timeseries[m, cur, j] = records[i][k][n]
                        cur += 1

                    elif diff_mins == 1:
                        continue
                    # TODO: могут ли быть другие случаи?
                    else:
                        break

                else:

                    timestamps.append(rec_time)
                    cur += 1
        
        # предупреждение о пробелах в данных
        if cur != self.reg_res:
            warn(f'Got only {cur} records from {filename}!')

        return timeseries

import matplotlib.pyplot as plt

loader = DataLoader('./data', 30)
for x, y in loader:
    fig, axs = plt.subplots(4)
    axs[0].imshow(x[:,:,0])
    axs[1].imshow(x[:,:,-1])
    axs[2].imshow(y[:,:,0])
    axs[3].imshow(y[:,:,-1])

    plt.show()