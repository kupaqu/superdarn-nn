class DataLoader:
    def __init__(self,
                 datapath,
                 shuffle=True):
        self.shuffle = shuffle
        self.data = {}

        for root, _, files in os.walk(datapath):
            for name in files:
                filename = name.split('.')
                self.data[filename[0] + filename[1]] = np.load(os.path.join(root, name))

    def __call__(self):
        keys = list(self.data.keys())
        if self.shuffle:
            random.shuffle(keys)
        # итерация по ключам в словаре self.data
        for key in keys:
            # если маска целевого значения пустая, то пропускаем пример
            if np.all(self.data[key][:,:,-1,:,:] == 0.):
                continue
            seq = self.__getSequence(key)
            arrays = []
            badCount = 0
            # итерация по историческим данным
            for item in seq:
                # некоторые исторические данные могут отсутствовать
                try:
                    if np.all(self.data[item][:,:,-1,:,:] == 0.):
                        badCount += 1
                    arrays.append(self.data[item])
                except KeyError:
                    # print(f'No key: {item}')
                    badCount += 1
                    arrays.append(np.zeros_like(self.data[key]))
            # если пропусков в данных больше чем 30%, то пропускаем пример
            # print(f'Bad count: {badCount}')
            if badCount / len(arrays) > 0.3:
                continue
            else:
                x = np.concatenate(arrays, axis=1)
                y = self.data[key]
                for beam in range(16):
                    yield x[:,:,:,beam,0], y[:,:,:-1,beam,0]
    
    def __getSequence(self, key):
        keyDT = datetime.strptime(key, '%Y%m%d%H%M')
        # список массивов периодов за неделю до целевого массива
        weekBefore = []
        for i in range(24*7, 0, -2):
            hoursBefore = (keyDT-timedelta(hours=i)).strftime('%Y%m%d%H%M')
            weekBefore.append(hoursBefore)
        return weekBefore
