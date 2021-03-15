import pandas as pd
import csv
import sys

from surprise import SVD
from surprise import Dataset
from surprise import Reader

def get_movies_names():
    with open('ml-100k/u.item', encoding = "ISO-8859-1") as csvfile:
        movies = list(csv.reader(csvfile, delimiter='|'))
    
    mov_name = {}
    
    for movie in movies:
        mov_name[movie[0]] = movie[1]
    
    return mov_name

def predict_ratings():
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file('ml-100k/u.data', reader=reader)
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    # Than predict ratings for all pairs (u, i) that are not in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    
    return predictions

def recommendations_for_user(usr):
    
    predictions = predict_ratings()
    
    df = pd.DataFrame()
    movies = get_movies_names()

    for user, item, rate, est, _ in predictions:
        if user == usr:
            df = df.append({"Movie":movies[item], "Estimate rating":float(est)}, ignore_index=True)

    df = df.set_index("Movie").sort_values(by=['Estimate rating'], ascending=False)
    print(df.head(5))

if __name__ == "__main__":
    try:
        recommendations_for_user(sys.argv[1])
    except:
        print("python movies_recommendations.py [user_id]")