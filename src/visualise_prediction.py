import pandas as pd
from pandas.core.frame import DataFrame
import utils
import joblib
import matplotlib.pyplot as plt



def run():
    data = utils.load_dataset()
    feature_columns = utils.get_feature_columns()
    scaler = joblib.load("./models/scaler_random.save")
    mlp = joblib.load("./models/mlp_random.save")

    data = data[data["CountryName"] == "Luxembourg"]

    # y_test = data['R']
    # x_test = data[feature_columns]
    # x_test = scaler.transform(x_test)
    # y_pred = mlp.predict(x_test)

    # df = DataFrame([y_pred.reshape(1,-1)], columns=["y_pred"])
    # df['y_test'] = data['R']
    # df['Date'] = data['Date']
    # data['y_pred'] = y_pred

    # data.plot(x='Date', y=["y_pred", "R"])
    data.set_index('Date', inplace=True)
    data.groupby('CountryName')['R'].plot(legend=True)
    plt.show()
    

if __name__ == '__main__':
    run()