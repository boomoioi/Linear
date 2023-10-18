import numpy as np
import tensorflow as tf
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import seaborn
yf.pdr_override()
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def predict(company, count, axis):
    df = pdr.get_data_yahoo(company, start=datetime.now() - timedelta(days=120), end=datetime.now())
    df = df.filter(["Close"])
    dataset = df.values
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    test_data = scaled_data[0: , :]
    
    x_test = []
    y_test = dataset[60:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    model = tf.keras.saving.load_model("my_model.keras")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    print()
    print("--------------------------------------------------------------------------")
    print()
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(rmse)
    valid = df[60:]
    valid['Predictions'] = predictions
    print(valid)
    axis[count].set_xlabel('Date', fontsize=18)
    axis[count].set_ylabel('Close Price USD ($)', fontsize=18)
    axis[count].plot(valid[['Close', 'Predictions']])
    axis[count].set_title(company) 

def correlation(com1, com2):
    stocks = [com1, com2]
    data = yf.download(stocks, start=datetime.now() - timedelta(days=10000), end=datetime.now())['Adj Close']

    corr_df = data.corr(method='pearson')
    corr_df.head().reset_index()
    print(corr_df)



figure, axis = plt.subplots(2, 1) 
con = "Y"
while con != "N":
    com1 = input("Enter first company : ")
    com2 = input("Enter second company : ")
    predict(com1, 0, axis)
    predict(com2, 1, axis)
    correlation(com1, com2)
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    con = input("Continue Y/N : ")