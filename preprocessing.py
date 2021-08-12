import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot(X, y, relation_type):

    colorset = {'black', 'red', 'green', 'blue', 'yellow',
                'navy', 'pink', 'gray', 'orange', 'lime',
                'gold', 'wheat', 'orchid', 'aqua', 'brown',
                'lawngreen', 'cornsilk', 'olive', 'indigo', 'hotpink', 'beige'}
    fig = plt.figure(figsize=(8, 6))
    # t = fig.suptitle(
    #     'Wine Residual Sugar - Alcohol Content - Acidity - Total Sulfur Dioxide - Type - Quality', fontsize=14)
    ax = fig.add_subplot(111, projection='3d')

    xs = list(map(float, X[:, 0]))
    ys = list(map(float, X[:, 1]))
    zs = list(map(float, X[:, 2]))

    data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]

    colors = []
    color_dict = {}

    for wt in list(y):
        if wt not in color_dict:
            color_dict[wt] = colorset.pop()
        colors.append(color_dict[wt])

    for data, color in zip(data_points, colors):
        x, y, z = data
        ax.scatter(x, y, z, c=color,
                   edgecolors='none')

    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    ax.set_zlabel('Day')

    plt.savefig('plt'+relation_type+'200.4.20.png')


def preprocessing_releasedate(datalist, parents, relation_type, k, block_size, trainXfilename, trainYfilename):
    # print("preprocessing_releasedate:", datalist.shape)
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

    # print(newdatalist[:150])
    # print(bin_data)

    bin_data.to_csv(trainXfilename, sep=",", index=False)
    pd.DataFrame(labels, columns=["label"]).to_csv(
        trainYfilename, sep=",", index=False)

    plot(newdatalist[:, 2:5], newdatalist[:, 5], relation_type)
    return bin_data, labels


def convert_listcol_to_dummies(data):
    df = pd.DataFrame(data, columns=['datalist'])
    df = df.datalist.apply(
        lambda x: x.split(',') if isinstance(x, str) else '0')
    bin_data = pd.get_dummies(df.apply(
        pd.Series).stack()).groupby(level=0).sum().astype(str)

    return bin_data


def preprocessing_IMDB(datalist, train_size, parents, relation_type, trainXfilename, trainYfilename):
    labels = []
    joinable_count = 0
    datalist = np.array(datalist)

    year = datalist[:, 2]

    df_year = pd.DataFrame(year, columns=["year"])
    bin_year = pd.get_dummies(df_year)

    bin_genre = convert_listcol_to_dummies(datalist[:, 3])
    bin_countries = convert_listcol_to_dummies(datalist[:, 8])
    bin_language = convert_listcol_to_dummies(datalist[:, 9])

    rating = datalist[:, 10]
    df_rating = pd.DataFrame(rating, columns=['rating'])
    df_rating = df_rating.rating.apply(
        lambda x: float(x.split('(')[0]) if isinstance(x, str) else 0)
    max_rate = df_rating.max()
    min_rate = df_rating.min()

    normalized_rating = (df_rating-min_rate)/(max_rate - min_rate)

    bin_data = pd.concat([bin_year, bin_genre, bin_countries,
                          bin_language, normalized_rating], axis=1)

    for row in datalist[:train_size]:
        # generate labels
        key = relation_type + str(row[0])
        if key not in parents:
            labels.append("unjoinable")
        else:
            joinable_count += 1
            labels.append(parents[key])

    bin_data.to_csv(trainXfilename, sep=",", index=False)
    pd.DataFrame(labels, columns=["label"]).to_csv(
        trainYfilename, sep=",", index=False)

    return bin_data, labels, max_rate, min_rate


def preprocessing_IMDB_test(datalist, max_rate, min_rate, train_bin_data):
    datalist = np.array(datalist)
    year = datalist[:, 2]

    df_year = pd.DataFrame(year, columns=["year"])
    bin_year = pd.get_dummies(df_year)

    bin_genre = convert_listcol_to_dummies(datalist[:, 3])
    bin_countries = convert_listcol_to_dummies(datalist[:, 8])
    bin_language = convert_listcol_to_dummies(datalist[:, 9])

    rating = datalist[:, 10]
    df_rating = pd.DataFrame(rating, columns=['rating'])
    df_rating = df_rating.rating.apply(
        lambda x: float(x.split('(')[0]) if isinstance(x, str) else 0)

    normalized_rating = (df_rating-min_rate)/(max_rate - min_rate)

    bin_data = pd.concat([bin_year, bin_genre, bin_countries,
                          bin_language, normalized_rating], axis=1)
    bin_data = bin_data.reindex(columns=train_bin_data.columns, fill_value=0)

    return bin_data


def preprocessing_OMDB(datalist, train_size, parents, relation_type, trainXfilename, trainYfilename):
    labels = []
    joinable_count = 0
    datalist = np.array(datalist)

    year = datalist[:, 2]

    df_year = pd.DataFrame(year, columns=["year"])
    bin_year = pd.get_dummies(df_year)

    bin_genre = convert_listcol_to_dummies(datalist[:, 5])
    bin_countries = convert_listcol_to_dummies(datalist[:, 16])
    bin_language = convert_listcol_to_dummies(datalist[:, 15])

    rating = datalist[:, 11]
    df_rating = pd.DataFrame(rating, columns=['rating'])
    df_rating = df_rating.rating.apply(
        lambda x: 0 if math.isnan(x) else x)
    max_rate = df_rating.max()
    min_rate = df_rating.min()
    print(max_rate, min_rate)

    normalized_rating = (df_rating-min_rate)/(max_rate - min_rate)

    bin_data = pd.concat([bin_year, bin_genre, bin_countries,
                          bin_language, normalized_rating], axis=1)

    for row in datalist[:train_size]:
        # generate labels
        key = relation_type + str(row[0])
        if key not in parents:
            labels.append("unjoinable")
        else:
            joinable_count += 1
            labels.append(parents[key])

    bin_data.to_csv(trainXfilename, sep=",", index=False)
    pd.DataFrame(labels, columns=["label"]).to_csv(
        trainYfilename, sep=",", index=False)

    return bin_data, labels, max_rate, min_rate


def preprocessing_OMDB_test(datalist, max_rate, min_rate):
    datalist = np.array(datalist)

    year = datalist[:, 2]

    df_year = pd.DataFrame(year, columns=["year"])
    bin_year = pd.get_dummies(df_year)

    bin_genre = convert_listcol_to_dummies(datalist[:, 5])
    bin_countries = convert_listcol_to_dummies(datalist[:, 16])
    bin_language = convert_listcol_to_dummies(datalist[:, 15])

    rating = datalist[:, 11]
    df_rating = pd.DataFrame(rating, columns=['rating'])
    df_rating = df_rating.rating.apply(
        lambda x: 0 if math.isnan(x) else x)

    normalized_rating = (df_rating-min_rate)/(max_rate - min_rate)

    bin_data = pd.concat([bin_year, bin_genre, bin_countries,
                          bin_language, normalized_rating], axis=1)

    return bin_data
