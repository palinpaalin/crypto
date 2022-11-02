from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import numpy as np
import pandas as pd


def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

"""
LSTM
"""


def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """
    df_exc = df.select_dtypes(exclude='object')
    return df_exc.div(df_exc.iloc[0]) - 1


def normalise_min_max(df):
    """ Normalise dataframe column-wise min/max. """
    return (df - df.min()) / (df.max() - df.min())


def extract_window_data(df, window_len=10, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of len `window_data`.

        :param window_len: Size of window
        :param zero_base: If True, the data in each window is normalised to reflect changes
            with respect to the first entry in the window (which is then always 0)
    """
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size=test_size)

    # extract window data
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)

    # extract targets
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def prediction(df):
    train, test = train_test_split(df, test_size=0.1)
    np.random.seed(42)

    # data params
    window_len = 7
    test_size = 0.1
    zero_base = True

    # model params
    lstm_neurons = 20
    epochs = 50
    batch_size = 4
    loss = 'mae'
    dropout = 0.25
    optimizer = 'adam'

    train, test, X_train, X_test, y_train, y_test = prepare_data(
        df, 'Close', window_len=window_len, zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    targets = test['Close'][window_len:]
    preds = model.predict(X_test).squeeze()

    preds = test['Close'].values[:-window_len] * (preds + 1)
    preds = pd.DataFrame(index=targets.index, data=preds)

    return preds