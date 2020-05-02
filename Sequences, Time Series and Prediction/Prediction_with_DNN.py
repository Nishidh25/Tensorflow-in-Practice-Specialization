# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:28:49 2020

@author: Nishidh Shekhawat
"""
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%203.ipynb#scrollTo=4sTTIOCbyShY
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """
    Generates a windowed dataset from data

    Parameters
    ----------
    series : TYPE
        DESCRIPTION.
    window_size : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    shuffle_buffer : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dataset = tf.data.Dataset.from_tensor_slices(series) # creates dataset from series 
    # if series is 0-9
    # 0
    # 1
    # 2
    # 3
    # 4
    # 5
    # 6
    # 7
    # 8
    # 9
    
    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)   # splits data into into windows
    
    # 0 1 2 3 4 
    # 1 2 3 4 5 
    # 2 3 4 5 6 
    # 3 4 5 6 7 
    # 4 5 6 7 8 
    # 5 6 7 8 9 
    
    dataset = dataset.flat_map(lambda window :window.batch(window_size + 1)) # flatten the data , easier to work with 

    # [0 1 2 3 4]
    # [1 2 3 4 5]
    # [2 3 4 5 6]
    # [3 4 5 6 7]
    # [4 5 6 7 8]
    # [5 6 7 8 9]
    
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1])) # shuffle buffer for speed and split data into data_x and labels_y
    
    # [0 1 2 3] [4]
    # [1 2 3 4] [5]
    # [2 3 4 5] [6]
    # [3 4 5 6] [7]
    # [4 5 6 7] [8]
    # [5 6 7 8] [9]
    
    dataset = dataset.batch(batch_size).prefetch(1) # divide into batches
    
    # x =  [[4 5 6 7] [2 3 4 5]]
    # y =  [[8] [6]]
    # x =  [[1 2 3 4] [0 1 2 3]]
    # y =  [[5] [4]]
    # x =  [[5 6 7 8] [3 4 5 6]]
    # y =  [[9] [7]]
    
    return dataset 



dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss="mse", optimizer=optimizer)

history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=1)

lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])


forecast = []

for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)

print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
    