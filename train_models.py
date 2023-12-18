import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
from copy import deepcopy as dc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import pydot
import graphviz
from IPython.display import clear_output
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tensorflow.keras.models import load_model
from PlotLearning import PlotLearning

import json

def evaluation_minmax(model, X_train, y_train, X_val, y_val, X_test, y_test, save_path=None):
    prediction_train = model.predict(X_train)
    prediction_val = model.predict(X_val)
    predction_test = model.predict(X_test)

    scores = {
        'Training Set': {
            'Mean Squared Error (MSE)': round(mean_squared_error(y_true=y_train, y_pred=prediction_train), 4),
            'Root Mean Squared Error (RMSE)': round(np.sqrt(mean_squared_error(y_true=y_train, y_pred=prediction_train)), 4),
            'R-squared (R2) score': round(r2_score(y_true=y_train, y_pred=prediction_train), 4),
            'Explained variance score': round(explained_variance_score(y_true=y_train, y_pred=prediction_train), 4),
        },
        'Validation Set': {
            'Mean Squared Error (MSE)': round(mean_squared_error(y_true=y_val, y_pred=prediction_val), 4),
            'Root Mean Squared Error (RMSE)': round(np.sqrt(mean_squared_error(y_true=y_val, y_pred=prediction_val)), 4),
            'R-squared (R2) score': round(r2_score(y_true=y_val, y_pred=prediction_val), 4),
            'Explained variance score': round(explained_variance_score(y_true=y_val, y_pred=prediction_val), 4),
        },
        'Test Set': {
            'Mean Squared Error (MSE)': round(mean_squared_error(y_true=y_test, y_pred=predction_test), 4),
            'Root Mean Squared Error (RMSE)': round(np.sqrt(mean_squared_error(y_true=y_test, y_pred=predction_test)), 4),
            'R-squared (R2) score': round(r2_score(y_true=y_test, y_pred=predction_test), 4),
            'Explained variance score': round(explained_variance_score(y_true=y_test, y_pred=predction_test), 4),
        }
    }

    # Print the scores
    for set_name, set_scores in scores.items():
        print(f'{set_name} statistics:')
        for metric, value in set_scores.items():
            print(f'{metric}: {value}')

    # Save scores to a file
    if save_path:
        with open(save_path, 'w') as file:
            json.dump(scores, file)


def plot_minmax(model, X_train, y_train, X_val, y_val, X_test, y_test,type):
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    predictions_test = model.predict(X_test)

    sets = [('Training Set', y_train, predictions_train),
            ('Validation Set', y_val, predictions_val),
            ('Test Set', y_test, predictions_test)]

    for set_name, actual, prediction in sets:
        plt.figure(figsize=(30, 8))
        plt.plot(np.arange(len(actual)), actual, label='Actual',color='green')
        plt.plot(np.arange(len(prediction)), prediction, label='Predicted', color='red')
        plt.title(f'{set_name} Predictions vs Actuals')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig(f'{set_name}_{type}__plot.png')
        

def plot_residuals_histogram(model, X_train, y_train, X_val, y_val, X_test, y_test,type, bins=20):
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    predictions_test = model.predict(X_test)

    residuals_train = y_train - predictions_train
    residuals_val = y_val - predictions_val
    residuals_test = y_test - predictions_test

    sets = [('Training Set', residuals_train.flatten()),
            ('Validation Set', residuals_val.flatten()),
            ('Test Set', residuals_test.flatten())]

    for set_name, residuals in sets:
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=bins, alpha=0.7, color='blue')
        percentiles = [5, 95]  # Adjust percentiles as needed
        for percentile in percentiles:
            plt.axvline(np.percentile(residuals, percentile), color='red', linestyle='--', linewidth=1, label=f'{percentile}th Percentile')
        plt.title(f'{set_name} Residuals Histogram')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'{set_name}_{type}_residuals.png')
        
def recursive_pred(model,interval,intital_start,mean, std):
    last_window = intital_start
    output = []
    for i in range(interval):
        prediction = model.predict(np.array([last_window[-7:]]))
        output.append(prediction[0][0])
        rate_of_change =((prediction[0][0]-((last_window[-1][0]*std[0])+mean[0]))-mean[1])/std[1]
        prediction = (prediction-mean[0])/std[0]
        last_window = np.concatenate((last_window,np.concatenate((prediction,np.array([[rate_of_change]])),axis=1)),axis=0)
    return np.array(output)        
    
def recursive_pred_evaluation(model, interval, initial_set,evaluation_set, mean, std,rang):
    outputs = []
    residual = np.array([])
    for i in range(rang):
        print(f'{i} out of {rang}')
        pred = recursive_pred(model, interval,initial_set[i], mean, std)
        outputs.append(pred)
        residual = np.concatenate((residual,np.array(pred) - evaluation_set[i:interval + i]))
    return np.array(outputs), residual

def sliding_window_eval(val_set, interval, rang):
    return np.array([val_set[i:i+interval] for i in range(rang)])


def plot_recursive_pred(model, interval, X_train, X_val, X_test,y_train, y_val, y_test, mean, std,rang,type,score_save_path):
    fore_train, red_train = recursive_pred_evaluation(model, interval, X_train, y_train, mean,std, rang)
    fore_val, red_val = recursive_pred_evaluation(model, interval, X_val, y_val, mean,std, rang)
    fore_test, red_test = recursive_pred_evaluation(model, interval, X_test, y_test, mean,std, rang)

    sets = [('Training Set 7 days forecast', y_train, fore_train),
            ('Validation Set 7 days forecast', y_val, fore_val),
            ('Test Set 7 days forecast', y_test, fore_test)]
    for set_name, actual, prediction in sets:
        plt.figure(figsize=(30, 8))
        plt.plot( actual[:interval+rang-1], label='Actual',color='green')
        plt.title(f'{set_name} Predictions vs Actuals')
        plt.xlabel('Index')
        plt.ylabel('Values')
        for i in range(prediction.shape[0]):
            plt.plot(np.arange(i,i +interval), prediction[i], label='Predicted', color='red')
            plt.title(f'{set_name} Predictions vs Actuals')
            if i == 0:
                plt.legend()
        plt.savefig(f'{set_name}_{type}_forecast_plot.png')
        
    sets2 = [('Training Set 7 days forecast Residual', red_train),
            ('Validation Set 7 days forecast Residual', red_val),
            ('Test Set 7 days forecast Residual', red_test)]
    
        
    for set_name, residuals in sets2:
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=500, alpha=0.7, color='blue')
        percentiles = [5, 95]  # Adjust percentiles as needed
        for percentile in percentiles:
            plt.axvline(np.percentile(residuals, percentile), color='red', linestyle='--', linewidth=1, label=f'{percentile}th Percentile')
        plt.title(f'{set_name} Residuals Histogram')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'{set_name}_{type}_.png')
 
        
    scores = {
        'Training Set': {
            'Mean Squared Error (MSE)': round(mean_squared_error(y_true=sliding_window_eval(y_train, interval, rang), y_pred=fore_train), 4),
            'Root Mean Squared Error (RMSE)': round(np.sqrt(mean_squared_error(y_true=sliding_window_eval(y_train, interval, rang), y_pred=fore_train)), 4),
            'R-squared (R2) score': round(r2_score(y_true=sliding_window_eval(y_train, interval, rang), y_pred=fore_train), 4),
            'Explained variance score': round(explained_variance_score(y_true=sliding_window_eval(y_train, interval, rang), y_pred=fore_train), 4),
        },
        'Validation Set': {
            'Mean Squared Error (MSE)': round(mean_squared_error(y_true=sliding_window_eval(y_val, interval, rang), y_pred=fore_val), 4),
            'Root Mean Squared Error (RMSE)': round(np.sqrt(mean_squared_error(y_true=sliding_window_eval(y_val, interval, rang), y_pred=fore_val)), 4),
            'R-squared (R2) score': round(r2_score(y_true=sliding_window_eval(y_val, interval, rang), y_pred=fore_val), 4),
            'Explained variance score': round(explained_variance_score(y_true=sliding_window_eval(y_val, interval, rang), y_pred=fore_val), 4),
        },
        'Test Set': {
            'Mean Squared Error (MSE)': round(mean_squared_error(y_true=sliding_window_eval(y_test, interval, rang), y_pred=fore_test), 4),
            'Root Mean Squared Error (RMSE)': round(np.sqrt(mean_squared_error(y_true=sliding_window_eval(y_test, interval, rang), y_pred=fore_test)), 4),
            'R-squared (R2) score': round(r2_score(y_true=sliding_window_eval(y_test, interval, rang), y_pred=fore_test), 4),
            'Explained variance score': round(explained_variance_score(y_true=sliding_window_eval(y_test, interval, rang), y_pred=fore_test), 4),
        }
    }

    # Print the scores
    for set_name, set_scores in scores.items():
        print(f'{set_name} statistics:')
        for metric, value in set_scores.items():
            print(f'{metric}: {value}')

    # Save scores to a file
    if score_save_path:
        with open(score_save_path, 'w') as file:
            json.dump(scores, file)

       

        
        

#Load data Max Data and add Rate of change percentile column
def load_data(file,col):
    df = pd.read_csv(file, parse_dates=True,index_col="DATE",usecols=['DATE',col], dtype=float)
    df = dc(df['1990':])
    df['T_RATE_OC'] = df[col].pct_change() * 100 
    df.dropna(inplace=True)
    print(df.info())
    return df

#Split data into input and label.
# lookback parameter is the look back period. e.g (lookback = 1) [[day1],[day2],[day3]] -> [day4]
def df_to_X_y_min_max(df, lookback):
  df = df.to_numpy()
  X = []
  y = []
  for i in range(len(df)-lookback):
    row = [r for r in df[i:i+lookback]]
    X.append(row)
    label = df[i+lookback][0]
    y.append(label)
  return np.array(X), np.array(y)

def train_val_test_split_minmax(X,y,train_ratio, val_ratio, test_ratio):
    if train_ratio + val_ratio + test_ratio != 1:
        print('Ratios does not add up to 1')
    else:
        dataset_length = len(X)
        X_train, y_train = X[:round(dataset_length*train_ratio)], y[:round(dataset_length*train_ratio)]
        X_val, y_val = X[round(dataset_length*(train_ratio)):round(dataset_length*(train_ratio+val_ratio))], y[round(dataset_length*(train_ratio)):round(dataset_length*(train_ratio+val_ratio))]
        X_test, y_test = X[round(dataset_length*(train_ratio+val_ratio)):], y[round(dataset_length*(train_ratio+val_ratio)):]
        print(f'X_train shape: {X_train.shape}')
        print(f'y_train shape: {y_train.shape}')
        print(f'X_val shape: {X_val.shape}')
        print(f'y_val shape: {y_val.shape}')
        print(f'X_test: {X_test.shape}')
        print(f'y_test shape: {y_test.shape}')
        return X_train, y_train, X_val, y_val, X_test, y_test

def standardizing_minmax(X_train, y_train, X_val, y_val, X_test, y_test, n_features):
    mean = [np.mean(X_train[:,:,i]) for i in range(n_features)]
    std = [np.std(X_train[:,:,i]) for i in range(n_features)]
    for i in range(n_features):
        X_train[:,:,i] = (X_train[:,:,i] - mean[i])/ std[i]
        X_test[:,:,i] = (X_test[:,:,i] - mean[i])/ std[i]
        X_val[:,:,i] = (X_val[:,:,i] - mean[i])/ std[i]
    return X_train, y_train, X_val, y_val, X_test, y_test,mean,std

def prepare_minmax(X_train):
    model = Sequential()
    model.add(InputLayer((X_train.shape[1],X_train.shape[2])))
    model.add(LSTM(72, return_sequences=True))
    # model2.add(Dense(32, activation = 'relu'))
    model.add(LSTM(72))
    model.add(Dense(72, activation = 'relu'))
    # model2.add(Dense(200, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    print(model.summary())
    return model

def train_minmax(model, X_train, y_train, X_val, y_val, save_name,n_run):
    for i in range(n_run):
        print('RUN ------------------------------>',i)
        model.compile(loss='mse',
                    optimizer=Adam(learning_rate=0.0001), 
                    metrics=[RootMeanSquaredError(),])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,callbacks=[PlotLearning()],steps_per_epoch=10)
    model.save(save_name + '.keras')
    return model

#Training model Max and evaluate
df = load_data('weatherLAX.csv','TMAX')
X_train, y_train, X_val, y_val, X_test, y_test,mean, std = standardizing_minmax(*train_val_test_split_minmax(*df_to_X_y_min_max(df,7),0.8,0.1,0.1),2) 
model = prepare_minmax(X_train)
model = train_minmax(model, X_train, y_train, X_val, y_val, 'max_model', 6)
evaluation_minmax(load_model('max_model.keras'), X_train, y_train, X_val, y_val, X_test, y_test, save_path='max_scores.json')
# Call the function with your model and data
plot_minmax(load_model('max_model.keras'), X_train[-600:], y_train[-600:], X_val[-600:], y_val[-600:], X_test[-600:], y_test[-600:],'max')
# Call the function with your model and data
plot_residuals_histogram(load_model('max_model.keras'), X_train, y_train, X_val, y_val, X_test, y_test,'max', bins=1000)
plot_recursive_pred(load_model('max_model.keras'),7,X_train, X_val, X_test, y_train, y_val, y_test, mean,std,1200,'max','7_days_forecast_score(max_model).json')

#Traub min model and evaluate
df = load_data('weatherLAX.csv','TMIN')
X_train, y_train, X_val, y_val, X_test, y_test,mean, std = standardizing_minmax(*train_val_test_split_minmax(*df_to_X_y_min_max(df,7),0.8,0.1,0.1),2) 
model = prepare_minmax(X_train)
model = train_minmax(model, X_train, y_train, X_val, y_val, 'min_model', 6)
evaluation_minmax(load_model('min_model.keras'), X_train, y_train, X_val, y_val, X_test, y_test, save_path='min_scores.json')
plot_minmax(load_model('min_model.keras'), X_train[-600:], y_train[-600:], X_val[-600:], y_val[-600:], X_test[-600:], y_test[-600:],'min')
plot_residuals_histogram(load_model('min_model.keras'), X_train, y_train, X_val, y_val, X_test, y_test,'min', bins=1000)
plot_recursive_pred(load_model('min_model.keras'),7,X_train, X_val, X_test, y_train, y_val, y_test, mean,std,1200,'min','7_days_forecast_score(min_model).json')
