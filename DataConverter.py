import os
from tqdm import tqdm
import numpy as np
import bz2
import pydarnio
import datetime
import warnings

import sys

import matplotlib.pyplot as plt

class DataConverter:
    def __init__(self, src_dir: str, dst_dir: str, nrang=70, reg_res=60, bmnum=16, chnum=1, keys=['pwr0', 'v', 'p_l', 'p_s', 'w_l', 'w_s', 'qflg']):
        self.src_dir = os.path.abspath(src_dir)
        self.dst_dir = os.path.abspath(dst_dir)
        self.src_prefix = len(self.src_dir) + len(os.path.sep)

        self.nrang = nrang
        self.reg_res = reg_res
        self.keys = keys

        self.bmnum = bmnum
        self.chnum = chnum

    def __copy_dir_tree(self):
        for root, dirs, _ in os.walk(self.src_dir):
            for dir in dirs:
                # индекс последнего разделителя относительно
                # полного пути до исходной папки
                dirpath = os.path.join(self.dst_dir, root[self.src_prefix:], dir)
                try:
                    os.mkdir(dirpath)
                except FileExistsError:
                    continue
    
    def convert(self):
        self.__copy_dir_tree()
        for root, _, files in os.walk(self.src_dir):
            if len(files) > 0:
                print(f'Converting FITACF\'s to NumPy arrays at {root}')
                for file in tqdm(files):

                    src_filename = os.path.join(root, file)
                    if not src_filename.endswith('fitacf.bz2'):
                        continue

                    dst_filename = os.path.join(self.dst_dir, root[self.src_prefix:], file)[:-len('fitacf.bz2')]
                    content = self.__get_content(src_filename)
                    np.save(dst_filename+'npy', content)
    
    def __get_content(self, filename):
        timeseries = np.zeros(shape=(self.nrang, self.reg_res, len(self.keys), self.bmnum, self.chnum))

        # открытие файла на чтение
        with bz2.open(filename) as fp:
            fitacf_stream = fp.read()
        reader = pydarnio.SDarnRead(fitacf_stream, True)
        records = reader.read_fitacf()

        for beam in range(self.bmnum):

            for channel in range(self.chnum):

                # список меток времени
                timestamps = []
                # счетчик, показывающий текущий индекс в массивах timeseries и mask
                cur = 0

                for i in range(len(records)):

                    # если номер луча и канала совпадают с нужным
                    if records[i]['bmnum'] == beam and records[i]['channel'] == channel:
                        
                        rec_time = datetime.datetime(year=records[i]['time.yr'],
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
                                        if m >= self.nrang:
                                            continue
                                        for j, k in enumerate(self.keys):
                                            timeseries[m, cur, j, beam, channel] = records[i][k][n]

                                cur += 1

                            elif diff_mins == 1:
                                continue

                            # TODO: могут ли быть другие случаи?
                            else:
                                break

                        else:
                            timestamps.append(rec_time)
                            cur += 1
                    
                    if cur >= self.reg_res:
                        break
                
                # предупреждение о пробелах в данных
                if cur != self.reg_res:
                    warnings.warn(f'Got only {cur} records from {filename} at beam {beam}, channel {channel}!')
        # plt.imshow(timeseries[:,:,0,0,0])
        # plt.show()
        return timeseries

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    params = sys.argv
    try:
        src = params[1]
        dst = params[2]
        nrang = params[3]
    except IndexError:
        print('usage: src_path dst_path nrang')
        exit()
    
    converter = DataConverter(src, dst, int(nrang))
    converter.convert()