import numpy as np
import pandas as pd


def preprocessing_releasedate(datalist, parents, relation_type):
    labels = []
    datalist = np.array(datalist)
    dates = []
    joinable_count = 0
    for row in datalist:
        dates.append(row[2].split("-"))
        # generate labels
        key = relation_type + str(row[0]) + "," + str(row[1])
        if key not in parents:
            labels.append("unjoinable")
        else:
            joinable_count += 1
            labels.append(parents[key])

    dates = np.array(dates)
    newdatalist = np.delete(datalist, 2, 1)
    # newdatalist = np.delete(newdatalist, 0, 1)
    newdatalist = np.concatenate((newdatalist, dates), 1)

    # one hot encoding
    df = pd.DataFrame(
        newdatalist, columns=["mov_id", "country_id", "year", "month", "day"])
    bin_data = pd.get_dummies(
        df[["country_id", "year", "month", "day"]].astype(str))

    newdatalist = np.concatenate((newdatalist, np.expand_dims(labels, 1)), 1)

    return bin_data, labels
