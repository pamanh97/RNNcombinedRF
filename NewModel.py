import math
import random
import gc
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.tsa.seasonal as snl
import statsmodels.api as sm
import statsmodels.tsa.arima.model as arima_model
import statsmodels.tsa.ar_model as ar_model
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.api as smt
#from statsmodels.tsa.ar_model import ARMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz  # with pydot
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

g=0

class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError("Method fit not implemented")

    def predict(self, X):
        raise NotImplementedError("Method predict not implemented")


class ARIMAModel(BaseModel):
    def __init__(self, order):
        self.order = order
        self.model = None

    def fit(self, X, y=None):
        self.model = ARIMA(X, order=self.order).fit()

    def predict(self, X):
        return self.model.forecast(steps=len(X))[0]


# ===============================
# Hàm và Lớp Hỗ Trợ
# ===============================

def remove_trend_and_seasonality(data, freq=24):
    """
    Loại bỏ xu hướng và tính mùa vụ bằng cách phân rã tập dữ liệu
    """
    decomposition = sm.tsa.seasonal_decompose(data, model='additive', period=freq)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    decomposed = [trend, seasonal, residual]
    residual.dropna(inplace=True)
    return decomposed

def add_trend_and_seasonality_residual(decomposed, residual):
    """
    Adds the trend and seasonality to a residual dataset previously decomposed
    :param decomposed:
    :param residual:
    :return:
    """
    return decomposed[0] + decomposed[1] + residual

def scale_minmax_range(data, min_val, max_val, min_scale, max_scale):
    """
    Scales a dataset using a minmax function with range
    :param data:
    :param min_val:
    :param max_val:
    :param min_scale:
    :param max_scale:
    :return:
    """
    return ((data - min_val) / (max_val - min_val)) * (max_scale - min_scale) + min_scale


def rescale_minmax_range(data, min_val, max_val, min_scale, max_scale):
    """
    Rescales back a dataset using a minmax function with range
    :param data:
    :param min_val:
    :param max_val:
    :param min_scale:
    :param max_scale:
    :return:
    """
    return (data - min_scale) / (max_scale - min_scale) * (max_val - min_val) + min_val


def difference_timeseries(data, interval=1):
    """
    Creates a differenced dataset
    :param data:
    :param interval:
    :return:
    """
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return diff


def get_rmse(test, predictions):
    """
    Gets the RMSE by comparing the test data and predictions
    :param test:
    :param predictions:
    :return:
    """
    return np.sqrt(((np.asanyarray(test) - np.asanyarray(predictions)) ** 2).mean())


# invert differenced value
def inverse_difference_timeseries(data, predicted_value):
    """
    Gets the inverse dataset that was previously differenced
    :param data:
    :param predicted_value:
    :return:
    """
    return predicted_value + data[-1]

from statsmodels.tsa.stattools import adfuller

def check_stationarity(data):
    result = adfuller(data, autolag='AIC')
    adf_statistic = result[0]
    p_value = result[1]
    # Thường, p-value nhỏ hơn 0.05 cho thấy dữ liệu có tính stacionary
    return p_value < 0.05

def difference_data(data, interval=1):
    differenced = data.diff(periods=interval).dropna()
    return differenced



# ===============================
# ARMA
# ===============================

def evaluate_models_arma(data, p_values, g, q_values, is_stationary):
    """
    Evaluate combinations of p and q values for an ARMA model
    :param data:
    :param p_values:
    :param q_values:
    :param is_stationary:
    :return:
    """

    data = data.astype('float64')
    best_rmse, best_cfg = float("inf"), None
    for p in p_values:
        for q in q_values:
            try:
                if p == 0 and q == 0:
                    continue
                rmse = evaluate_model_arma(data, (1, 0, 1), is_stationary)
                if rmse < best_rmse:
                    best_rmse, best_cfg = rmse, (p, g, q)
            except:
                continue
    print('Best ARMA%s  RMSE=%.3f' % (best_cfg, best_rmse))
    return best_cfg


def evaluate_model_arma(data, order, is_stationary, verbose=1):
    """
    Evaluate an ARMA model for a given order (p, g, q) and return RMSE
    :param X:
    :param arma_order:
    :param is_stationary:
    :param verbose:
    :return:
    """
    order = (1, 0, 1)
    if order is None:
        raise ValueError("The 'order' argument is None. Please provide a valid (p, g, q) tuple.")


        # Assuming 'order' is a tuple in the form (p, g, q)
      # This line extracts p and q from the 'order' tuple
    # Kiểm tra tính stacionary nếu tham số is_stationary là None
    if is_stationary is None:
        is_stationary = check_stationarity(data)
    if not is_stationary:
        print("Dữ liệu không stacionary, áp dụng differencing")
        data = difference_data(data)

    start = time.perf_counter()  # Hoặc time.process_time() tùy thuộc vào yêu cầu  # Start Timer
    # prepare training dataset
    data = data.astype('float64')
    train_size = int(len(data) * 0.90)
    train, test = data[0:train_size], data[train_size:]
    train_data = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        stationary_data = train_data
        # decomposed = remove_trend_and_seasonality(train)
        # If it's not stationary, difference or decompose first
        if not is_stationary:
            stationary_data = difference_timeseries(train_data, interval=1)
            # stationary_data = decomposed[2]
        p=1
        q=1
        model = smt.ARIMA(stationary_data, order=(p, g, q))  # for ARMA(p, q) behavior
        model_fit = model.fit()
        predicted_value = model_fit.forecast()[0][0]
        # inverse the difference if not stationary
        if not is_stationary:
            predicted_value = inverse_difference_timeseries(train_data, predicted_value)
            # predicted_value = add_trend_and_seasonality_residual(decomposed, predicted_value)

        # print('inverse', predicted_value)

        predictions.append(predicted_value)
        train_data.append(test[t])

    end = time.perf_counter() - start  # Hoặc time.process_time() - start  # End Timer

    # print(predictions)
    # calculate RMSE
    rmse = get_rmse(predictions, test)
    print('ARMA%s RMSE:%.3f Time_Taken:%.3f' % (order, rmse, end))

    if verbose == 1:
        print(f'ARIMA{order} RMSE: {rmse:.3f} Time Taken: {end:.3f}s')
        print(f'Predictions: {predictions}')

    return rmse


# ===============================
# ARIMA
# ===============================

def evaluate_auto_arima(data, seasonal=False, m=1):
    """
    Automatically evaluates the ARIMA model using Auto ARIMA to find the best p, d, q parameters
    :param data: The dataset for which to model the ARIMA process
    :param seasonal: Whether the ARIMA model should consider seasonality
    :param m: The number of periods in each season (if seasonal is True)
    :return: The best fit ARIMA model
    """
    auto_arima_model = auto_arima(data, seasonal=seasonal, m=m, suppress_warnings=True,
                                  error_action="ignore", stepwise=True)
    print(f"Best ARIMA Model: {auto_arima_model.order}, Seasonal Order: {auto_arima_model.seasonal_order}")
    return auto_arima_model

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models_arima(data, p_values, d_values, q_values, is_stationary):
    """
    Evaluate combinations of p, d and q values for an ARIMA model
    :param data:
    :param p_values:
    :param d_values:
    :param q_values:
    :param is_stationary:
    :return:
    """

    # Kiểm tra tính stacionary nếu tham số is_stationary là None
    if is_stationary is None:
        is_stationary = check_stationarity(data)
    if not is_stationary:
        print("Dữ liệu không stacionary, áp dụng differencing")
        data = difference_data(data)

    data = data.astype('float64')
    best_rmse, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    if p == 0 and q == 0:
                        continue
                    rmse = evaluate_model_arima(data, (p, d, q), is_stationary)
                    if rmse < best_rmse:
                        best_rmse, best_cfg = rmse, (p, d, q)
                except:
                    continue
    print('Best ARIMA%s  RMSE=%.3f' % (best_cfg, best_rmse))
    return best_cfg


def evaluate_model_arima(data, order, is_stationary, verbose=1):
    """
    Evaluate an ARIMA model for a given order (p,d,q) and return RMSE
    :param X:
    :param arma_order:
    :param is_stationary:
    :param verbose:
    :return:
    """
    start = time.perf_counter()  # Start Timer
    # prepare training dataset
    data = data.astype('float64')
    train_size = int(len(data) * 0.90)
    train, test = data[0:train_size], data[train_size:]
    train_data = [x for x in train]
    # make predictions
    model = None
    predictions = list()
    for t in range(len(test)):
        stationary_data = train_data
        # decomposed = remove_trend_and_seasonality(train)
        # If it's not stationary, difference or decompose first
        if not is_stationary:
            stationary_data = difference_timeseries(train_data, interval=1)
            # stationary_data = decomposed[2]

        model = ARIMA(stationary_data, order=order)
        model_fit = model.fit(disp=0)
        predicted_value = model_fit.forecast()[0][0]
        # print('predicted_value1', model_fit.forecast()[0])
        # print('predicted_value2', predicted_value)
        # inverse the difference if not stationary
        if not is_stationary:
            predicted_value = inverse_difference_timeseries(train_data, predicted_value)
            # predicted_value = add_trend_and_seasonality_residual(decomposed, predicted_value)

        predictions.append(predicted_value)
        train_data.append(test[t])

    end = time.perf_counter() - start  # End Timer
    # print(predictions)
    # calculate RMSE
    rmse = get_rmse(predictions, test)
    print('ARIMA%s RMSE:%.3f  Time_Taken:%.3f' % (order, rmse, end))

    if verbose == 1:
        print(f'ARIMA{order} RMSE: {rmse:.3f} Time Taken: {end:.3f}s')
        print(f'Predictions: {predictions}')

    return rmse

# Ví dụ sử dụng ARIMA với API mới
def fit_arima_model(data):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())

# Ví dụ xây dựng một mô hình LSTM đơn giản
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Ví dụ sử dụng RandomForestRegressor với cách tiếp cận mới
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    print(f"Độ chính xác trên tập kiểm tra: {model.score(X_test, y_test)}")
# ===============================
# Hyperparameters
# ===============================

train_size_percentage = 0.9  # Training size
mutation_rate = 0.1  # Mutation rate for GA
min_mutation_momentum = 0.0001  # Min mutation momentum
max_mutation_momentum = 0.1  # Max mutation momentum
min_population = 20  # Min population for GA
max_population = 50  # Max population for GA
num_Iterations = 3  # Number of iterations to evaluate GA
look_back = 1  # Num of timespaces to look back for training and testing
max_dropout = 0.2  # Maximum percentage of dropout
min_num_layers = 1  # Min number of hidden layers
max_num_layers = 10  # Max number of hidden layers
min_num_neurons = 10  # Min number of neurons in hidden layers
max_num_neurons = 100  # Max number of neurons in hidden layers
min_num_estimators = 100  # Min number of random forest trees
max_num_estimators = 500  # Max number of random forest trees
force_gc = True  # Forces garbage collector
rnn_epochs = 1  # Epochs for RNN

# ===============================
# Constants and variables
# ===============================

datasets = [
    'datacsv/Air Temperature.csv',
    'datacsv/Chlorophyll Fluorescence.csv',
    'datacsv/Depth.csv',
    'datacsv/Dissolved Oxygen.csv',
    'datacsv/PAR.csv',
    'datacsv/Pressure.csv',
    'datacsv/Rain Fall.csv',
    'datacsv/Relative Humidity.csv',
    'datacsv/Salinity.csv',
    'datacsv/Sonde pH.csv',
    'datacsv/Turbidity.csv',
    'datacsv/Water Density.csv',
    'datacsv/Water Temperature.csv',
    'datacsv/Wind Direction.csv',
    'datacsv/Wind Speed.csv'
]
optimisers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
rnn_types = ['LSTM', 'GRU', 'SimpleRNN']


# fix random seed for reproducibility
# np.random.seed(0)

def create_dataset(dataset, look_back=1):
    """
    Converts an array of values into a dataset matrix
    :param dataset:
    :param look_back:
    :return:
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    collect_gc()

    return np.array(dataX), np.array(dataY)


def collect_gc():
    """
    Forces garbage collector
    :return:
    """
    if force_gc:
        gc.collect()


def load_dataset(dataset_path):
    """
    Loads a dataset with training and testing arrays
    :param dataset_path:
    :return:
    """
    # Load dataset
    dataset = pd.read_csv(dataset_path, parse_dates=True, index_col=0)
    dataset = dataset.values  # as numpy array
    dataset = dataset.astype('float64')
    # Normalise the dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * train_size_percentage)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    train_x_stf = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x_stf = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    train_x_st = np.reshape(train_x, (train_x.shape[0], 1))
    test_x_st = np.reshape(test_x, (test_x.shape[0], 1))

    return dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y


def generate_rnn(hidden_layers):
    """
    Generates a RNN using an array of hidden layers including the number of neurons for each layer
    :param hidden_layers:
    :return:
    """
    # Create and fit the RNN
    model = Sequential()
    # Add input layer
    model.add(Dense(1, input_shape=(1, look_back)))

    # Add hidden layers
    for i in range(len(hidden_layers)):
        neurons_layer = hidden_layers[i]
        # Randomly select rnn type of layer
        rnn_type_index = random.randint(0, len(rnn_types) - 1)
        rnn_type = rnn_types[rnn_type_index]

        dropout = random.uniform(0, max_dropout)  # dropout between 0 and max_dropout
        return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking

        # Select and add type of layer
        if rnn_type == 'LSTM':
            model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        elif rnn_type == 'GRU':
            model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        elif rnn_type == 'SimpleRNN':
            model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))

    collect_gc()

    # Add output layer
    model.add(Dense(1))
    return model


def evaluate_rnn(model, train_x, test_x, train_y, test_y, scaler, optimiser):
    """
    Evaluates the RNN model using the training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :param optimiser:
    :return:
    """
    model.compile(loss='mean_squared_error', optimizer=optimiser)
    model.fit(train_x, train_y, epochs=rnn_epochs, batch_size=1, verbose=2)
    # Forecast
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # Invert forecasts
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    # print('Train Score: %.2f RMSE' % (train_score))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    # print('Test Score: %.2f RMSE' % (test_score))
    model.train_score = train_score
    model.test_score = test_score

    return train_score, test_score, train_predict, test_predict


def crossover_rnn(model_1, model_2):
    """
    Executes crossover for the RNN in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # new_model = copy.copy(model_1)
    new_model = model_1

    # Probabilty of models depending on their RMSE test score
    # Lower RMSE score has higher prob
    test_score_total = model_1.test_score + model_2.test_score
    model_1_prob = 1 - (model_1.test_score / test_score_total)
    model_2_prob = 1 - model_1_prob
    # Probabilities of each item for each model (all items have same probabilities)
    model_1_prob_item = model_1_prob / (len(model_1.layers) - 2)
    model_2_prob_item = model_2_prob / (len(model_2.layers) - 2)

    # Number of layers of new generation depend on probability of each model
    num_layers_new_gen = int(model_1_prob * (len(model_1.layers) - 1) + model_2_prob * (len(model_2.layers) - 1))

    # Create list of int with positions of the layers of both models.
    cross_layers_pos = []
    # Create list of weights
    weights = []
    # Add positions of layers for model 1. Input and ouput layer are not added.
    for i in range(2, len(model_1.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 1
        cross_layers_pos.append(mod_item)
        weights.append(model_1_prob_item)

    # Add positions of layers for model 2. Input and ouput layer are not added.
    for i in range(2, len(model_2.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 2
        cross_layers_pos.append(mod_item)
        weights.append(model_2_prob_item)

    collect_gc()

    # If new num of layers are larger than the num crossover layers, keep num of crossover layers
    if num_layers_new_gen > len(cross_layers_pos):
        num_layers_new_gen = len(cross_layers_pos)

    # Randomly choose num_layers_new_gen layers of the new list
    cross_layers_pos = list(np.random.choice(cross_layers_pos, size=num_layers_new_gen, replace=False, p=weights))

    # Add both group of hidden layers to new group of layers using previously chosen layer positions of models
    cross_layers = []
    for i in range(len(cross_layers_pos)):
        mod_item = cross_layers_pos[i]
        if mod_item.model == 1:
            cross_layers.append(model_1.layers[mod_item.pos])
        else:
            cross_layers.append(model_2.layers[mod_item.pos])

    collect_gc()

    # Add input layer randomly from parent 1 or parent 2
    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.insert(0, model_1.layers[0])
    else:
        cross_layers.insert(0, model_2.layers[0])

    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.append(model_1.layers[len(model_1.layers) - 1])
    else:
        cross_layers.append(model_2.layers[len(model_2.layers) - 1])

    # Set new layers
    new_model.layers = cross_layers

    return new_model


def mutate_rnn(model):
    """
    Mutates the RNN model
    :param model:
    :return:
    """
    for i in range(len(model.layers)):
        # Mutate randomly each layer
        bit_random = random.uniform(0, 1)

        if bit_random <= mutation_rate:
            weights = model.layers[i].get_weights()  # list of weights as numpy arrays
            # calculate mutation momentum
            mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
            new_weights = [x * mutation_momentum for x in weights]
            model.layers[i].set_weights(new_weights)

    collect_gc()


def save_plot_model_rnn(model):
    """
    Saves the plot of the RNN model
    :param model:
    :return:
    """
    plot_model(model, show_shapes=True)


def generate_rf(estimators):
    """
    Generates a Random Forest with the number of estimators to use
    :param estimators:
    :return:
    """
    # Create and fit the RF
    model = RandomForestRegressor(n_estimators=estimators, criterion='mse', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                  oob_score=False, n_jobs=1, random_state=None, verbose=0)

    return model


def evaluate_rf(model, train_x, test_x, train_y, test_y, scaler):
    """
    Evaluates the Random Forest with training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :return:
    """

    model.fit(train_x, train_y)
    # Forecast
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # Invert forecasts
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:]))
    # print('Train Score: %.2f RMSE' % (train_score))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:]))
    # print('Test Score: %.2f RMSE' % (test_score))
    model.train_score = train_score
    model.test_score = test_score

    return train_score, test_score, train_predict, test_predict


def crossover_rf(model_1, model_2):
    """
    Executes crossover for the RF in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # new_model = copy.copy(model_1)
    new_model = model_1

    # Probabilty of models depending on their RMSE test score
    test_score_total = model_1.test_score + model_2.test_score
    model_1_prob = 1 - model_1.test_score / test_score_total
    model_2_prob = 1 - model_1_prob

    # New estimator is the sum of both estimators times their probability
    new_model.n_estimators = math.ceil(model_1.n_estimators * model_1_prob + model_2.n_estimators * model_2_prob)

    return new_model


def mutate_rf(model):
    """
    Mutates the Random Forest
    :param model:
    :return:
    """
    # Mutate randomly the estimator
    bit_random = random.uniform(0, 1)

    if bit_random <= mutation_rate:
        # calculate mutation momentum
        mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
        # Mutate estimators
        model.n_estimators = model.n_estimators + math.ceil(model.n_estimators * mutation_momentum)


def save_plot_model_rf(model):
    """
    Saves the plot of the Random Forest model
    :param model:
    :return:
    """
    for i in range(len(model.estimators_)):
        estimator = model.estimators_[i]
        out_file = open("trees/tree-" + str(i) + ".dot", 'w')
        export_graphviz(estimator, out_file=out_file)
        out_file.close()


def ensemble_stacking(model_1_values, model_2_values, test, scaler):
    """
    Ensemble result of 2 models using stacking and averaging.
    Takes both model predictions, averages them and calculates the new RMSE
    :param model_1_values:
    :param model_2_values:
    :return:
    """
    # Generates the stacking values by averaging both predictions
    stacking_values = []
    for i in range(len(model_1_values)):
        stacking_values.append((model_1_values[i][0] + model_2_values[i]) / 2)

    test = scaler.inverse_transform([test])
    rmse = math.sqrt(mean_squared_error(test[0], stacking_values))
    return stacking_values, rmse


def evaluate_ga(dataset):
    """
    Evaluates and generates the ensemble model using Genetic Algorithms
    :param dataset:
    :return:
    """
    print('#-----------------------------------------------')
    print('  ', dataset)
    print('#-----------------------------------------------')

    dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y = load_dataset(dataset)
    start = time.perf_counter()  # Start Timer
    num_population = random.randint(min_population, max_population)  # Number of RNN to evaluate
    # == 1) Generate initial population for RNN and Random Forest
    population_rnn = []
    population_rf = []
    start_ga_1 = time.perf_counter()  # Start Timer
    for i in range(num_population):
        # -- RNN
        # Generate random topology configuration
        num_layers = random.randint(min_num_layers, max_num_layers)
        hidden_layers = []
        for j in range(num_layers):
            num_neurons = random.randint(min_num_neurons, max_num_neurons)
            hidden_layers.append(num_neurons)

        collect_gc()

        # Generate and add rnn model to population
        model_rnn = generate_rnn(hidden_layers)
        population_rnn.append(model_rnn)

        # -- RF
        # Generate random number of estimators for RF
        num_estimators = random.randint(min_num_estimators, max_num_estimators)

        # Generate and add rf model to population
        model_rf = generate_rf(num_estimators)
        population_rf.append(model_rf)

    end_ga_1 = time.perf_counter() - start_ga_1  # End Timer
    print('Generate Initial population Time_Taken:%.3f' % end_ga_1)

    collect_gc()
    # print(len(population))

    best_rmse_rnn = float("inf")
    best_rmse_rf = float("inf")
    best_rnn_model = None
    best_test_predict_rnn = None
    best_rf_model = None
    best_test_predict_rf = None
    # Evaluate fitness for
    for i in range(num_Iterations):
        print('=================================================================================================')
        print(' iteration: %d, total iterations: %d, population size: %d ' % (i + 1, num_Iterations, num_population))
        print('=================================================================================================')
        # train_score, test_score = float("inf"), float("inf")
        # == 2)  Evaluate fitness for population
        start_ga_2 = time.perf_counter()  # Start Timer
        for j in range(num_population):
            # Evaluate fitness for RNN
            rnn_model = population_rnn[j]
            train_score_rnn, test_score_rnn, train_predict_rnn, test_predict_rnn = evaluate_rnn(rnn_model, train_x_stf,
                                                                                                test_x_stf, train_y,
                                                                                                test_y, scaler,
                                                                                                optimisers[0])
            # print('test predictions RNN: ', test_predict_rnn)
            print('test_score RMSE RNN:%.3f ' % test_score_rnn)

            if test_score_rnn < best_rmse_rnn:
                best_rmse_rnn = test_score_rnn
                # best_rnn_model = copy.copy(rnn_model)
                best_rnn_model = rnn_model
                best_test_predict_rnn = test_predict_rnn

            # Evaluate fitness for RF
            rf_model = population_rf[j]
            train_score_rf, test_score_rf, train_predict_rf, test_predict_rf = evaluate_rf(rf_model, train_x_st,
                                                                                           test_x_st, train_y, test_y,
                                                                                           scaler)
            # print('test predictions RF: ', test_predict_rf)
            print('test_score RMSE RF:%.3f ' % test_score_rf)

            if test_score_rf < best_rmse_rf:
                best_rmse_rf = test_score_rf
                # best_rf_model = copy.copy(rf_model)
                best_rf_model = rf_model
                best_test_predict_rf = test_predict_rf

        end_ga_2 = time.perf_counter() - start_ga_2  # End Timer
        print('Evaluate Fitness population Time_Taken:%.3f' % end_ga_2)

        collect_gc()

        print('Temporal Best RMSE RNN:%.3f' % best_rmse_rnn)
        print('Temporal Best predictions: ', [x[0] for x in best_test_predict_rnn])
        print('Temporal Best RMSE RF:%.3f' % best_rmse_rf)
        print('Temporal Best predictions: ', [x for x in best_test_predict_rf])

        # == 3) Create new population with new generations
        # Every generation will use the current best RNN and best RF to mate
        start_ga_3 = time.perf_counter()  # Start Timer
        for pop_index in range(num_population):
            # Select parents for mating
            # Element at pop_index as parent. This will be replaced with the new generation
            rnn_model_1 = population_rnn[pop_index]
            rf_model_1 = population_rf[pop_index]
            # 2 parent is the best found so far
            rnn_model_2 = best_rnn_model
            rf_model_2 = best_rf_model

            # == 4) Create new generation with crossover
            new_rnn_model = crossover_rnn(rnn_model_1, rnn_model_2)
            new_rf_model = crossover_rf(rf_model_1, rf_model_2)

            # == 5) Mutate new generation
            mutate_rnn(new_rnn_model)
            mutate_rf(new_rf_model)

            # Replace current model in population
            population_rnn[pop_index] = new_rnn_model
            population_rf[pop_index] = new_rf_model

        end_ga_3 = time.perf_counter() - start_ga_3  # End Timer
        print('Generate new population Time_Taken:%.3f' % end_ga_3)

        collect_gc()

    collect_gc()

    end = time.perf_counter() - start  # End Timer

    print('=============== BEST RNN ===============')
    print('Best predictions: ', [x[0] for x in best_test_predict_rnn])
    print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rnn, end))
    save_plot_model_rnn(best_rnn_model)

    print('=============== BEST RF ===============')
    print('Best predictions: ', [x for x in best_test_predict_rf])
    print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rf, end))

    save_plot_model_rf(best_rf_model)
    # print(best_rf_model.get_params(deep=True))

    # Ensemble
    print('=============== Ensemble ===============')
    averaging_values, rmse = ensemble_stacking(best_test_predict_rnn, best_test_predict_rf, test_y, scaler)
    print('Ensemble averaging_values: ', averaging_values)
    print('Ensemble rmse: ', rmse)


def evaluate_bptt(dataset):
    """
    Evaluates and generates a RNN model using BPTT
    :param dataset:
    :return:
    """
    print('#-----------------------------------------------')
    print('  ', dataset)
    print('#-----------------------------------------------')

    dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y = load_dataset(dataset)
    start = time.perf_counter()  # Start Timer

    # Generate a 1 hidden layer configuration
    hidden_layers = [1]
    # Generate and add rnn model to population
    model_rnn = generate_rnn(hidden_layers)

    train_score_rnn, test_score_rnn, train_predict_rnn, test_predict_rnn = evaluate_rnn(model_rnn, train_x_stf,
                                                                                        test_x_stf, train_y,
                                                                                        test_y, scaler,
                                                                                        optimisers[0])

    end = time.perf_counter() - start  # End Timer

    print('Predictions: ', [x[0] for x in test_predict_rnn])
    print('RMSE:%.3f Time_Taken:%.3f' % (test_score_rnn, end))
    save_plot_model_rnn(model_rnn)

# ===============================
# Main Execution
# ===============================

def main():
    """
    Chính thức thực thi các quá trình đánh giá và huấn luyện mô hình
    """
    datasets = [
        'datacsv/Air Temperature.csv',
        'datacsv/Chlorophyll Fluorescence.csv',
        'datacsv/Depth.csv',
        'datacsv/Dissolved Oxygen.csv',
        'datacsv/PAR.csv',
        'datacsv/Pressure.csv',
        'datacsv/Rain Fall.csv',
        'datacsv/Relative Humidity.csv',
        'datacsv/Salinity.csv',
        'datacsv/Sonde pH.csv',
        'datacsv/Turbidity.csv',
        'datacsv/Water Density.csv',
        'datacsv/Water Temperature.csv',
        'datacsv/Wind Direction.csv',
        'datacsv/Wind Speed.csv'
    ]

    # Đánh giá mô hình ARMA và ARIMA cho từng dataset
    for dataset_path in datasets:
        print(f'Processing dataset: {dataset_path}')
        dataset = pd.read_csv(dataset_path, parse_dates=True, index_col=0)
        dataset = dataset[dataset.columns[0]].dropna()

        p_values = range(0, 3)
        d_values = range(1, 3)
        q_values = range(0, 3)

        best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
        best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)

        print(f'Best ARMA configuration: {best_arma}')
        print(f'Best ARIMA configuration: {best_arima}')

        # Thực hiện đánh giá GA và BPTT nếu cần
    """
        Main execution
        :return:
        """
    datasets = [
        'datacsv/Air Temperature.csv',
        'datacsv/Chlorophyll Fluorescence.csv',
        'datacsv/Depth.csv',
        'datacsv/Dissolved Oxygen.csv',
        'datacsv/PAR.csv',
        'datacsv/Pressure.csv',
        'datacsv/Rain Fall.csv',
        'datacsv/Relative Humidity.csv',
        'datacsv/Salinity.csv',
        'datacsv/Sonde pH.csv',
        'datacsv/Turbidity.csv',
        'datacsv/Water Density.csv',
        'datacsv/Water Temperature.csv',
        'datacsv/Wind Direction.csv',
        'datacsv/Wind Speed.csv'
    ]

    # load dataset
    print('#-----------------------------------------------')
    print('  ', datasets[0])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[0], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[0]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[1])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[1], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[0]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, False)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, False)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, False, 1)
    evaluate_model_arima(dataset, best_arima, False, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[2])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[2], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[0]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[3])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[3], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[3]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[4])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[4], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[4]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[5])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[5], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[5]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[6])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[6], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[6]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[7])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[7], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[7]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[8])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[8], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[8]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[9])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[9], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[9]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[10])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[10], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[10]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[11])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[11], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[11]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[12])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[12], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[12]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[13])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[13], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[13]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)

    print('#-----------------------------------------------')
    print('  ', datasets[14])
    print('#-----------------------------------------------')
    dataset = pd.read_csv(datasets[14], parse_dates=True, index_col=0)
    dataset = dataset[dataset.columns[14]]
    # evaluate parameters
    p_values = range(0, 3)
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    dataset.dropna(inplace=True)
    # Find best arma, arima
    best_arma = evaluate_models_arma(dataset, p_values, g, q_values, True)
    best_arima = evaluate_models_arima(dataset, p_values, d_values, q_values, True)
    # Get predicted values for best arma, arima
    print('=============== BEST ===============')
    evaluate_model_arma(dataset, best_arma, True, 1)
    evaluate_model_arima(dataset, best_arima, True, 1)


    evaluate_ga(datasets[0])
    evaluate_ga(datasets[1])
    evaluate_ga(datasets[2])
    evaluate_ga(datasets[3])
    evaluate_ga(datasets[4])
    evaluate_ga(datasets[5])
    evaluate_ga(datasets[6])
    evaluate_ga(datasets[7])
    evaluate_ga(datasets[8])
    evaluate_ga(datasets[9])
    evaluate_ga(datasets[10])
    evaluate_ga(datasets[11])
    evaluate_ga(datasets[12])
    evaluate_ga(datasets[13])
    evaluate_ga(datasets[14])

    evaluate_bptt(datasets[0])
    evaluate_bptt(datasets[1])
    evaluate_bptt(datasets[2])
    evaluate_bptt(datasets[3])
    evaluate_bptt(datasets[4])
    evaluate_bptt(datasets[5])
    evaluate_bptt(datasets[6])
    evaluate_bptt(datasets[7])
    evaluate_bptt(datasets[8])
    evaluate_bptt(datasets[9])
    evaluate_bptt(datasets[10])
    evaluate_bptt(datasets[11])
    evaluate_bptt(datasets[12])
    evaluate_bptt(datasets[13])
    evaluate_bptt(datasets[14])

if __name__ == "__main__":
    main()
