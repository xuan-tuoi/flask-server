# -- coding: utf-8
# This is a sample Python script.
from KMeansModel import KmeansModel
from RecommenderModel import RecommenderModel
from crudFeatureCsv import *


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # 1. Train Kmeans phân cụm người dùng, 1 lần/ngày hoặc 1 lần/tuần
    km = KmeansModel('csvData/history_orders.csv')
    km.preprocessing()
    km.train(5)
    km.predict()

    # 2. Dự đoán thể loại mà người dùng thích dựa vào phân cụm người dùng
    # truyền vào id người dùng và số lượng thể loại mong muốn
    # trả về danh sách n thể loại mà cụm người dùng đó nghe nhiều nhất
    res = km.recommend_genre_of_user_by_id('3eacb1fc-d0d1-44ee-adb0-a058db492343', 4)
    # for r in res:
    #     print('r::::', r)
    # res = km.recommend_genre_of_user_by_id(1000, 2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
