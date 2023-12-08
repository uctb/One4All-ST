import datetime
import numpy as np
from dateutil.parser import parse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from ..preprocess import MoveSample, SplitData, ST_MoveSample, chooseNormalizer
import os
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import numpy as np

class WeatherDataset():
    def __init__(self,dataset,city,data_dir='data') -> None:
        #! 我们默认weather dataset里面的值进行了时间对齐以及填充等
        '''
        {
            TimeRange: [start_date_str,end_date_str]
            TimeFitness: int(in minutes)
            WeatherValue: pd.DateFrame with shape (num_timeslots,num_dimension)
            WeatherInfo: list with num_dimension elements, one of the element is [name,type]
        }
        '''
        self.task = dataset
        self.city = city
        data_path = os.path.join(data_dir,'{}_{}.pkl'.format(dataset,city))
        with open(data_path,'rb') as fp:
            self.dataset = pkl.load(fp)
        df = self.dataset['WeatherValue']
        weather_info = self.dataset['WeatherInfo']
        self.time_range = self.dataset['TimeRange']
        self.time_fitness = self.dataset['TimeFitness']
        continuous_value = []
        discrete_value = []
        for name,type in weather_info.items():
            #TODO: 如果是str则进行one hot 编码
            #TODO: 如果是float|int则直接转成numpy.ndarray
            if type == 'str' or type == 'int':
                enc = OrdinalEncoder()
                enc.fit(np.array((df[name].unique())).reshape([-1,1]))
                tmp_value = enc.transform(np.array(df[name]).reshape([-1,1]))
                discrete_value.append(tmp_value)
            else:
                tmp_value = np.array(df[name]).reshape([-1,1])
                continuous_value.append(tmp_value)
        self.weather_c_value = np.concatenate(continuous_value,axis=1)
        self.weather_d_value = np.concatenate(discrete_value,axis=1)

class WeatherContextLoader():
    # 完成空维复制和采样两个功能
    def __init__(self,data_dir,dataset,city,data_range,test_ratio,train_data_length,history_len,normalize,target_len,num_stations) -> None:
        self.dataset = WeatherDataset(dataset=dataset,city=city,data_dir=data_dir)
        self.daily_slots = int(24 * 60 / self.dataset.time_fitness)
        self.history_len = history_len
        if type(data_range) is str and data_range.lower().startswith("0."):
            data_range = float(data_range)
        if type(data_range) is str and data_range.lower() == 'all':
            data_range = [0, len(self.dataset.weather_c_value)]
        elif type(data_range) is float:
            data_range = [0, int(data_range * len(self.dataset.weather_c_value))]
        else:
            data_range = [int(data_range[0] * self.daily_slots), int(data_range[1] * self.daily_slots)]

        #? 可能会有点占内存 spatial duplicate
        continuous_data = self.dataset.weather_c_value[data_range[0]:data_range[1],:].astype(np.float32)
        discrete_data = self.dataset.weather_d_value[data_range[0]:data_range[1],:].astype(np.float32)

        self.continuous_data = np.tile(continuous_data[:,np.newaxis,:],[1,num_stations,1])
        self.discrete_data = np.tile(discrete_data[:,np.newaxis,:],[1,num_stations,1])
        
        if test_ratio > 1 or test_ratio < 0:
            raise ValueError('test_ratio ')
        self.train_test_ratio = [1 - test_ratio, test_ratio]

        self.train_c_data, self.test_c_data = SplitData.split_data(self.continuous_data, self.train_test_ratio)
        self.train_d_data, self.test_d_data = SplitData.split_data(self.discrete_data, self.train_test_ratio)
        
        # Normalize continuous weather data

        self.normalizer = chooseNormalizer(normalize,self.train_c_data)
        self.train_c_data = self.normalizer.transform(self.train_c_data)
        self.test_c_data = self.normalizer.transform(self.test_c_data)

        if train_data_length.lower() != 'all':
            train_day_length = int(train_data_length)
            self.train_c_data = self.train_c_data[-int(train_day_length * self.daily_slots):]
            self.train_d_data = self.train_d_data[-int(train_day_length * self.daily_slots):]
        
        train_data_len = len(self.train_c_data)

        expand_start_index = train_data_len - self.history_len

        self.test_c_data = np.vstack([self.train_c_data[expand_start_index:], self.test_c_data])
        self.test_d_data = np.vstack([self.train_d_data[expand_start_index:], self.test_d_data])

        test_data_len = len(self.test_c_data)
        
        # build train feature



        win_size = history_len + target_len

        num_feature = train_data_len - win_size + 1

        self.train_sequence_length = num_feature

        c_feature = []
        d_feature = []
        for i in range(num_feature):
            c_feature.append([self.train_c_data[win_size+i-target_len-1-step*1] for step in range(history_len)])
            d_feature.append([self.train_d_data[win_size+i-target_len-1-step*1] for step in range(history_len)])
        self.train_history_c_feature = np.transpose(np.array(c_feature),[0,2,1,3])
        self.train_history_d_feature = np.transpose(np.array(d_feature),[0,2,1,3])
        self.train_current_c_weather = self.train_c_data[-num_feature:]
        self.train_current_d_weather = self.train_d_data[-num_feature:]



        win_size = history_len + target_len

        num_feature = test_data_len - win_size + 1

        self.test_sequence_length = num_feature


        c_feature = []
        d_feature = []
        for i in range(num_feature):
            c_feature.append([self.test_c_data[win_size+i-target_len-1-step*1] for step in range(history_len)])
            d_feature.append([self.test_d_data[win_size+i-target_len-1-step*1] for step in range(history_len)])
        self.test_history_c_feature = np.transpose(np.array(c_feature),[0,2,1,3])
        self.test_history_d_feature = np.transpose(np.array(d_feature),[0,2,1,3])
        self.test_current_c_weather = self.test_c_data[-num_feature:]
        self.test_current_d_weather = self.train_d_data[-num_feature:]


if __name__ == '__main__':
    dataset = WeatherDataset('Weather',city='Chicago')