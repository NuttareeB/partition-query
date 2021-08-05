import numpy as np
import pandas as pd


def preprocessing_releasedate(datalist, parents):
    labels = []

    datalist = np.array(datalist)
    dates = []
    for row in datalist:
        dates.append(row[2].split("-"))
    dates = np.array(dates)
    newdatalist = np.delete(datalist, 2, 1)
    # newdatalist = np.delete(newdatalist, 0, 1)
    newdatalist = np.concatenate((newdatalist, dates), 1)

    df = pd.DataFrame(
        newdatalist, columns=["mov_id", "country_id", "year", "month", "day"])
    bin_data = pd.get_dummies(
        df[["country_id", "year", "month", "day"]].astype(str))

    return bin_data
