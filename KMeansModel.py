# -- coding: utf-8
import csv
import random
import warnings

import pandas as pd
import requests
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class KmeansModel:
    def __init__(self, user_history_data_path):
        self.filepath = user_history_data_path
        self.df = pd.read_csv(user_history_data_path)

# Mist 8c
    def preprocessing(self):
        gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2, 0: 0, 1:1, 2:2}  # Ánh xạ giá trị chuỗi thành số
        self.df.gender = self.df.gender.map(gender_mapping)
        # print(self.df.head())
        self.user_id_col = self.df['user_id']

        self.df = self.df.drop(columns=['user_id'])
        if 'cluster' in self.df.columns:
            self.df = self.df.drop(columns=['cluster'])

        self.df = self.df.dropna() # loại bỏ giá trị null
        self.df = self.df.drop_duplicates() # loại bỏ giá trị bị trùng lặp

    def train(self, n_clusters):
        self.X = self.df.iloc[:, 0:22]  # 1t for rows and second for columns
        cols = self.X.columns
        min_max_scaler = preprocessing.MinMaxScaler() # chuẩn hóa dữ liệu
        np_scaled = min_max_scaler.fit_transform(self.X)

        # # new data frame with the new scaled data.
        self.X = pd.DataFrame(np_scaled, columns=cols)
        # Sau khi tiền xử lí dữ liêuj thì sẽ thu giảm số chiều
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.X)
        header = ''
        for i in range(1, 4):
            header += f' pca_{i}'
        header = header.split()
        self.X = pd.DataFrame(X_pca, columns=header)
        print(self.X.head())
        warnings.simplefilter('ignore')

        self.kmeans = KMeans(n_clusters)
        self.kmeans.fit(self.X)

    def predict(self):
        # print(self.user_id_col)
        identified_clusters = self.kmeans.fit_predict(self.X)
        self.df.insert(loc=22, column='cluster', value=identified_clusters)
        self.df.insert(loc=23, column='user_id', value=self.user_id_col)
        self.df.to_csv('csvData/predict.csv', index=False)
        print(self.df.head())

    def predict_v2(self):
        print(self.df.head())
        # print(self.user_id_col)
        identified_clusters = self.kmeans.fit_predict(self.X)
        self.df.insert(loc=22, column='cluster', value=identified_clusters)
        self.df.insert(loc=23, column='user_id', value=self.user_id_col)
        return self.df

    def recommend_genre_of_user_by_id(self, user_id, number_of_genres):
        # df1 = pd.read_csv(self.filepath)
        self.df = pd.read_csv('csvData/predict.csv')
        df3 = self.df.loc[self.df['user_id'] == user_id]
        results = []
        print(self.df.shape, df3.shape)
        if df3.shape[0] == 0:
            print('This is a new user and his history does not have much to explore')
            return results
        user_cluster = df3.iloc[0]['cluster']
        print('This user belongs to cluster ', user_cluster)
        df2 = self.df.loc[self.df['cluster'] == user_cluster]
        sorted_views = df2.iloc[:, 2:22].sum().sort_values(ascending=False)
        for x, y in sorted_views.items():
            print(x, y)
        i = 0
        for x, y in sorted_views.items():
            results.append(x)
            print('Recommend products from category:"', x, '" for user with id:', user_id)
            i += 1
            if i == number_of_genres:
                break
        return results

    def generate_features(self):
        header = 'userId blues classical country disco hiphop jazz metal pop reggae rock'
        header = header.split()
        file = open(self.filepath, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for i in range(1, 501):
            to_append = f'{i}'
            for j in range(0, 10):
                random_float = random.uniform(0.0, 1.0)
                if random_float < 0.4:
                    to_append += f' {random.randint(0, 10)}'
                elif random_float > 0.6:
                    to_append += f' {random.randint(80, 100)}'
                else:
                    to_append += f' {random.randint(10, 80)}'
            file = open(self.filepath, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        self.df = pd.read_csv(self.filepath)

    def updateListHistoryOrder(self):
        # URL của API
        api_url = "https://cosmetic-ecommerce.onrender.com/api/v1/users/history"
        # Gửi GET request và nhận dữ liệu từ API
        response = requests.get(api_url)
        # Kiểm tra xem request có thành công hay không
        if response.status_code == 200:
            # Lấy dữ liệu JSON từ response
            data = response.json()
            # # Đường dẫn tới tệp CSV bạn muốn tạo
            # csv_file_path = self.filepath
            csv_file_path= self.filepath

            # # Mở tệp CSV để ghi
            with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
                # Tạo đối tượng CSV writer
                csv_writer = csv.writer(csv_file)

                # Viết header (nếu có)
                # Ví dụ: Giả sử data là một danh sách các dictionaries và chúng có các key giống nhau
                if data:
                    header = data[0].keys()
                    csv_writer.writerow(header)

                # Viết dữ liệu từ danh sách dictionaries vào tệp CSV
                for row in data:
                    csv_writer.writerow(row.values())

                print(f"Dữ liệu đã được lưu vào {csv_file_path}")
        else:
            print("Error:", response.status_code, response.text)
