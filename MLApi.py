# -- coding: utf-8
from flask import Flask, jsonify, request, send_file
import os

from KMeansModel import KmeansModel
# from KnnModel import KnnModel
from RecommenderModel import *
from crudFeatureCsv import *

app = Flask(__name__)
app.config["UPLOAD_DIR"] = "Data/unlabeled/"
# knn = KnnModel('MLmodels/minMaxScaler.pkl', 'MLmodels/pca_21.pkl', 'MLmodels/knn1.joblib')
km = KmeansModel('csvData/predict.csv')
km2 = KmeansModel('csvData/history_orders.csv')


@app.route('/hello', methods=['GET'])
def helloworld():
    if request.method == 'GET':
        return send_file('Data\\tempSong\\temp.mp3')
    # data = {"data": "Hello World"}
    # return jsonify(data)


@app.route("/recommend-for-user", methods=["GET"])
def recommend_product_by_user_id():
    if request.method == 'GET':
        user_id = (request.args.get('userId'))
        print(user_id)
        number_of_genres = int(request.args.get('n'))
        print(number_of_genres)
        if number_of_genres > 10 or number_of_genres < 1:
            number_of_genres = 2
        print(user_id)
        res = km.recommend_genre_of_user_by_id(user_id, number_of_genres)
        data = {"data": {}}
        for index, value in enumerate(res):
            data['data'].update({"category" + str(index): value})
            print(value)
        return jsonify(data)


@app.route("/trainUserKmeans", methods=["GET"])
def train_user_kmeans():
    if request.method == "GET":
        km2.updateListHistoryOrder()
        km2.preprocessing()
        km2.train(5)
        res = km2.predict()
        data = {"mess": "trained successfully"}
        return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
