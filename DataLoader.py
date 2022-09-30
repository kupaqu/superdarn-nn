import os
import numpy as np
import datetime as dt
import glob
from warnings import warn
import bz2
import pydarn

class DataLoader:
    def __init__(self,
                    datapath: str, # путь к данным в формате fitacf
                    nrang: int, # количество гейтов
                    keys: list, # каналы данных
                    total_bms: int, # количество лучей
                    total_chs: int, # количество каналов
                    delta_hours: int = 2, # сдвиг по времени в fitacf файлах
                    reg_res: int = 60, # регулярное разрешение в данных (обычно 2 минуты)
                    windows_num: int = 30 # количество периодов
                    ):
        self.datapath = datapath
        self.nrang = nrang
        self.keys = keys
        self.total_bms = total_bms
        self.total_chs = total_chs
        self.delta_hours = delta_hours
        self.reg_res = reg_res
        self.windows_num = windows_num
    
    def __iter__(self):
        for radar_subdir, _, _ in os.walk(self.datapath): # итерация по подпапкам данных каждого из радаров
            for hour in range(0, 24, self.delta_hours): # итерация по периодам
                filenames = self.__get_filenames(radar_subdir, hour) # вытаскиваем все имена файлов за период указаный в переменной hour
                for i in range(len(filenames)-self.windows_num+1): # итерация скользящим окном по файлам
                    for channel in range(self.total_chs): # итерация по номерам каналов
                        for beam in range(self.total_bms): # итерация по номерам лучей
                            yield self.__get_data(filenames=filenames[i:(i+self.windows_num)], # извлечение данных в виде многомерного массива
                                                    bmnum=beam,
                                                    channel=channel)
    
    def __get_filenames(self, directory: str, hour: int):
        formatted_time = dt.time(hour=hour, minute=0)
        filenames = sorted(glob.glob(directory+f'/*.{formatted_time.strftime("%H%M")}.00.*.fitacf.bz2'))
        patches = [] # массив для пропущенных данных
        # проверка остутствия пропусков во временном ряде
        if filenames:
            sliding_date = self.__get_date_from_filename(filenames[0])
            delta = dt.timedelta(days=1)
            for i in range(len(filenames)):
                while self.__get_date_from_filename(filenames[i]) != sliding_date:
                    warn(f'WARNING: There is no data for {hour} hours 0 minutes at {sliding_date} in {directory}')
                    patches.append(f"{directory}/{sliding_date.strftime('%Y%m%d')}.patch")
                    sliding_date += delta
                sliding_date += delta
            filenames += patches
        return sorted(filenames)

    def __get_date_from_filename(self, filename):
        return dt.datetime.strptime(filename.split('/')[-1].split('.')[0], '%Y%m%d')

    def __get_data(self,
                    filenames: list,
                    bmnum: int,
                    channel: int):
        timeseries = np.zeros(shape=(self.nrang, 0, len(self.keys))) # сплошной массив для конкатенации всех периодов
        mask = np.zeros(shape=(self.nrang, 0, 1)) # сплошной массив для конкатенации всех масок

        for filename in filenames:
            if filename.endswith('patch'):
                timeseries = np.concatenate((timeseries,
                                            np.zeros(shape=(self.nrang, self.reg_res, len(self.keys)))), axis=1)
                mask = np.concatenate((mask,
                                            np.zeros(shape=(self.nrang, self.reg_res, 1))), axis=1)
            else:
                with bz2.open(filename) as fp:
                    fitacf_stream = fp.read()
                reader = pydarn.SuperDARNRead(fitacf_stream, True)
                records = reader.read_fitacf()

                if self.nrang != records[0]['nrang']:
                    warn('WARNING: The specified nrang does not match the nrang in the file!')

                aux_timeseries = np.zeros(shape=(self.nrang, 0, len(self.keys)))
                aux_mask = np.zeros(shape=(self.nrang, 0, 1))

                for i in range(len(records)):
                    if records[i]['bmnum'] == 0 and records[i]['channel'] == 0:
                        line = np.zeros(shape=(self.nrang, 1, len(self.keys)))
                        mask_line = np.zeros(shape=(self.nrang, 1, 1))
                        try:
                            for n, m in enumerate(records[i]['slist']):
                                for j, key in enumerate(self.keys):
                                    line[m, 0, j] = records[i][key][n]
                                mask_line[m, 0] = records[i]['qflg'][n]

                            aux_timeseries = np.concatenate((aux_timeseries, line), axis=1)
                            aux_mask = np.concatenate((aux_mask, mask_line), axis=1)

                        except KeyError:

                            aux_timeseries = np.concatenate((aux_timeseries, line), axis=1)
                            aux_mask = np.concatenate((aux_mask, mask_line), axis=1)

                if self.reg_res != aux_timeseries.shape[1]:
                    # TODO: как интерполировать значения от n>60 до 60?
                    warn(f'The specified regular resolution does not match the regular in {filename}, which is {aux_timeseries.shape[1]}')
                
                timeseries = np.concatenate((timeseries, aux_timeseries), axis=1)
                mask = np.concatenate((mask, aux_mask), axis=1)

                print(f'timeseries shape:{timeseries.shape}')
                print(f'mask shape:{mask.shape}\n')

        return (timeseries, mask)        

loader = DataLoader('./data', 75, ['p_l', 'pwr0', 'v'], 16, 1)
for file in loader:
    pass