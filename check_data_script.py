import os
import bz2
import numpy as np
import pydarn
import matplotlib.pyplot as plt

fitacf_file = os.path.dirname(__file__)+'/data/20220814.0200.00.sas.a.fitacf.bz2'

with bz2.open(fitacf_file) as fp:
  fitacf_stream = fp.read()
reader = pydarn.SuperDARNRead(fitacf_stream, True)
records = reader.read_fitacf()

pydarn.RTP.plot_range_time(records, beam_num=0, parameter='p_l', channel=0,
                            range_estimation=pydarn.RangeEstimation.RANGE_GATE,
                            colorbar_label='SNR')
plt.show()

nrang = records[0]['nrang']
key = 'p_l'

timeseries = np.zeros(shape=(nrang, 0))

print(f'Length of records: {len(records)}')
for i in range(len(records)):
  # if 'slist' not in records[i]:
  #   continue
  if records[i]['bmnum'] == 0 and records[i]['channel'] == 0:
    line = np.zeros(nrang)
    for n, m in enumerate(records[i]['slist']):
      line[m] = records[i][key][n]
    timeseries = np.concatenate((timeseries, line.reshape((-1,1))), axis=1)
    print(f"{records[i]['time.hr']}:{records[i]['time.mt']}:{records[i]['time.sc']}")

print(f'Timeseries shape: {timeseries.shape}')
plt.rcParams["figure.figsize"] = (5,5)
plt.imshow(np.rot90(timeseries.T, k=1))
plt.show()