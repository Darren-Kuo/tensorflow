import tensorflow as tf
from pymongo import MongoClient
from datetime import datetime as dt

from LSTM_Model import LSTM_Model
from Backtesting import Backtesting
from Data_Processing import Data_Processing


DB_URL = ''
DB_NAME = ''
COL_NAME = ''

D_FROM = dt.strptime('2017-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
D_TO = dt.strptime('2019-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
LAST_PROFIT = 0

LR = 0.005
TRAINING_ITERS = 2000
BATCH_SIZE = 500

N_STEPS = 20
N_INPUTS = 11
N_HID_UNITS = 20
N_OUTPUTS = 3
FORGET_BIAS = 1

MODEL_NAME = './txf_1hr_lstm.ckpt'

data_process = Data_Processing()
client = MongoClient(DB_URL)[DB_NAME][COL_NAME]
    
data = data_process.Get_Data(D_FROM, D_TO, '1H', client)
data['ma5'] = data.Close.rolling(5).mean()
data['ma10'] = data.Close.rolling(10).mean()
data['ma15'] = data.Close.rolling(15).mean()
data['ma20'] = data.Close.rolling(20).mean()
data['ma40'] = data.Close.rolling(40).mean()
data['ma60'] = data.Close.rolling(60).mean()

_data = data.tail(-60)

start = '2017-09-01'
end = '2018-12-31'
mask = (_data.index > start) & (_data.index <= end)
df_train = _data.loc[mask]

start = '2019-01-01'
end = '2019-06-01'
mask = (_data.index > start) & (_data.index <= end)
df_test = _data.loc[mask]

norm_train = data_process.Normalization(df_train)
norm_test = data_process.Normalization(df_test)
lab_train = data_process.Get_Label(df_train)
lab_test = data_process.Get_Label(df_test)

train_x, train_y = data_process.Build_Train(norm_train, lab_train)
test_x, test_y = data_process.Build_Train(norm_test, lab_test)

size = len(train_x)

### training & testing ###
test_loss = []
test_accuracy = []

for i in range(TRAINING_ITERS):
    if i == 0:
        lstm_train = LSTM_Model(N_STEPS, N_INPUTS, N_HID_UNITS, N_OUTPUTS, FORGET_BIAS, LR)
    else:
        lstm_train = LSTM_Model(N_STEPS, N_INPUTS, N_HID_UNITS, N_OUTPUTS, FORGET_BIAS, LR, isRestore = True, model_name = MODEL_NAME)
        
    ### prepare data ###
    train_x, train_y = data_process.Shuffle_Data(train_x, train_y)
    
    tf_train_x = tf.convert_to_tensor(train_x, tf.float32)
    tf_train_y = tf.one_hot(train_y, 3)
    
    _train_x = []
    _train_y = []
    for b in range(size // BATCH_SIZE):
        _train_x.append(tf_train_x[b * BATCH_SIZE:(b+1) * BATCH_SIZE])
        _train_y.append(tf_train_y[b * BATCH_SIZE:(b+1) * BATCH_SIZE])
    
    train_l, train_acc = lstm_train.Training(_train_x, _train_y)
    
    lstm_train.Save_Model(MODEL_NAME)
    if i % 50 == 0:
        print(train_l, train_acc)
        ### convert data to tensor & one hot ###
        tf_test_x = tf.convert_to_tensor(test_x, tf.float32)
        tf_test_y = tf.one_hot(test_y, 3)
    
        l, acc = lstm_train.Testing(tf_test_x, tf_test_y)
        test_loss.append(l)
        test_accuracy.append(acc)
        
        backtest = Backtesting(lstm_train)
        profit = backtest.Backtest(df_test.head(-20), tf_test_x)
        print('profit from backtest is %f' % profit)
        
        if profit > LAST_PROFIT:
            model_name = './txf_1hr_lstm_%s.ckpt' % i
            detail_name = './detail_txf_1hr_lstm_%s.csv' % i
            lstm_train.Save_Model(model_name)
            backtest.Save_Detail(detail_name)
            
            LAST_PROFIT = profit
    
    lstm_train.Close()
            
    
print('-------------------')
print(test_loss)
print('-------------------')
print(test_accuracy)