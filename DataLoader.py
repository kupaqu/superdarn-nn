import os
import glob
import datetime as dt
import bz2
import pydarn
import numpy as np

class DataLoader:
    def __init__(self, datapath: str, nrang: int, keys: list, total_bms: int, total_chs: int,
                        delta_hours: int = 2, regular_resolution: int = 120, window_number: int = 30):
        self.datapath = datapath

        self.total_bms = total_bms
        self.total_chs = total_chs
        self.nrang = nrang
        self.keys = keys
        
        self.delta_hours = delta_hours
        self.regular_resolution = regular_resolution
        self.window_number = window_number
    
    def __iter__(self):
        for subdir, _, _ in os.walk(self.datapath):
            for hour in range(0, 24, self.delta_hours):
                filenames = self.__get_filenames(subdir, hour)
                for i in range(len(filenames)-self.window_number+1):
                    # yield filenames[i:(i+self.window_number)]
                    # TODO: написать алгоритм возвращающий numpy массив
                    for channel in range(self.total_chs):
                        for beam in range(self.total_bms):
                            yield self.__truncate(filenames[i:(i+self.window_number)], bmnum=beam, channel=channel)
    
    def __truncate(self, filenames: list, bmnum, channel):
        timeseries = np.zeros(shape=(self.nrang, 0, len(self.keys)))
        for fitacf_file in filenames:
            with bz2.open(fitacf_file) as fp:
                fitacf_stream = fp.read()
            reader = pydarn.SuperDARNRead(fitacf_stream, True)
            records = reader.read_fitacf()
            if self.nrang != records[0]['nrang']:
                print('WARNING: The specified nrang does not match the nrang in the file!')
            
            aux_timeseries = np.zeros(shape=(self.nrang, 0, len(self.keys)))
            for i in range(len(records)):
                if records[i]['bmnum'] == 0 and records[i]['channel'] == 0:
                    line = np.zeros(shape=(self.nrang, 1,len(self.keys)))
                    for n, m in enumerate(records[i]['slist']):
                        for j, key in enumerate(self.keys):
                            line[m, 0, j] = records[i][key][n]
                    aux_timeseries = np.concatenate((aux_timeseries, line), axis=1)
                    # print(f"{records[i]['time.hr']}:{records[i]['time.mt']}:{records[i]['time.sc']}")
            print(self.regular_resolution == aux_timeseries.shape[1])
            if self.regular_resolution != aux_timeseries.shape[1]:
                raise Exception(f'The specified regular resolution does not match the regular in {fitacf_file}, which is {len(records)}')
            
            timeseries = np.concatenate((timeseries, aux_timeseries), axis=1)

            print(timeseries.shape)
                
    def __get_filenames(self, directory: str, hour: int):
        formatted_time = dt.time(hour=hour, minute=0)
        filenames = sorted(glob.glob(directory+f'/*.{formatted_time.strftime("%H%M")}.00.*.fitacf.bz2'))

        # проверка остутствия пропусков во временном ряде
        if filenames:
            sliding_date = self.__get_date_from_filename(filenames[0])
            delta = dt.timedelta(days=1)
            for i in range(len(filenames)):
                while self.__get_date_from_filename(filenames[i]) != sliding_date:
                    print(f'WARNING: There is no data for {hour} hours 0 minutes at {sliding_date} in {directory}')
                    sliding_date += delta
                sliding_date += delta

        return filenames
            
    def __get_date_from_filename(self, filename):
        return dt.datetime.strptime(filename.split('/')[-1].split('.')[0], '%Y%m%d')
        
loader = DataLoader('./data', 75, ['p_l', 'pwr0', 'v'], 16, 1)
for file in loader:
    break