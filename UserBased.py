import math
import statistics
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate
from sklearn.model_selection import KFold



def read_dataset(path):
    ds = []
    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = int(parts[2])

            ds.append((user_id, movie_id, rating))

    return ds


def user_prediction(user, movie, ratings, neighbors):
    dividend = sum([n[1] * (ratings[n[0]][movie] - statistics.mean(ratings[n[0]].values())) for n in neighbors])
    divisor = sum([n[1] for n in neighbors])

    try:
        return statistics.mean(ratings[user].values()) + (dividend / divisor)
    except ZeroDivisionError:
        return statistics.mean(ratings[user].values())

def present(model, knn, results):
    rows = []
    for i in range(len(results)):
        rows.append([model, knn, i+1, results[i]])
    rows.append([model, knn, "Average", statistics.mean(results)])

    print(tabulate(rows, headers=["Model", "KNN", "Fold", "MAE"]))


def pearson_correlation(u1, u2):
    mean_u1 = statistics.mean(u1.values())
    mean_u2 = statistics.mean(u2.values())

    commons = set(u1.keys()).intersection(set(u2.keys()))

    dividend = sum([(u1[c] - mean_u1) * (u2[c] - mean_u2) for c in commons])
    dvr1 = math.sqrt(sum([(u1[c] - mean_u1) ** 2 for c in commons]))
    dvr2 = math.sqrt(sum([(u2[c] - mean_u2) ** 2 for c in commons]))

    divisor = dvr1 * dvr2

    try:
        return dividend / divisor
    except ZeroDivisionError:
        return 0


def user_based(train, test, knn):
    ratings, similarities = defaultdict(lambda: dict()), defaultdict(lambda: dict())
    truth, predictions = [], []

    for user_id, movie_id, rating in train:
        ratings[user_id][movie_id] = rating

    for user_id, movie_id, rating in test:
        truth.append(rating)

        others = [k for k in ratings.keys() if k != user_id]
        for o in others:
            if o not in similarities[user_id]:
                similarities[user_id][o] = pearson_correlation(u1=ratings[user_id], u2=ratings[o])

        relative = [i for i in similarities[user_id].items() if movie_id in ratings[i[0]]]
        nearest = sorted(relative, key=lambda temp: temp[1], reverse=True)[:knn]

        p = user_prediction(user=user_id, movie=movie_id, ratings=ratings, neighbors=nearest)
        predictions.append(p)
    return mean_absolute_error(truth, predictions)

if __name__ == '__main__':
    path1 = input("Please enter the path to the data file:")
    data = read_dataset(path1)
    n1 = int(input("Please enter the number of kfold: (5 or 10)"))
    kf = KFold(n_splits=n1)
    knn1 = int(input("Please enter the number of k nearest neighbors(choises :10,20,30,40,50,60,70,80)"))
    maes = []
    for train_index, text_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in text_index]


        mae = user_based(train_data, test_data, knn1)
        maes.append(mae)


    present(results=maes, knn=knn1, model="user")

