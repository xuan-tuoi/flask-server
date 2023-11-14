from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import requests
import csv


class RecommendModel:
    def __init__(self, features_csv_path):
        self.filepath = features_csv_path
        self.df = pd.read_csv(features_csv_path)
        self.productId = []

    def preprocessing(self):
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        self.productId = self.df['id']
        self.df = self.df.drop(columns=['id', 'product_name'])

        shop_mapping = {'40c9cb9c-1628-4995-b8a6-7a5ce183e087': 1, '8c4451cf-3617-4d71-b003-c09d2eb5aa53': 2, '9a420be2-4df3-4595-a500-2211de5a9701': 3}  # Ánh xạ giá trị chuỗi thành số
        self.df.user_id = self.df.user_id.map(shop_mapping)

        categories = ['Eye cream', 'Exfoliator', 'Peptide', 'Cleansing oil', 'Sun cream', 'Cleanser', 'Cream', 'Face Mask', 'Sunscreen', 'Bath Salts', 'Body Wash', 'HA',
              'Toner', 'Mask', 'Mist', 'Body Lotion', 'Gel', 'Moisturizer', 'Hair', 'Serum']

        # Tạo một đối tượng LabelEncoder và áp dụng cho cột 'category'
        label_encoder = LabelEncoder()
        # Áp dụng LabelEncoder để chuyển đổi danh mục thành số
        categories_encoded = label_encoder.fit_transform(categories)
        self.df['product_category'] = label_encoder.fit_transform(self.df['product_category'])

        self.X = self.df.iloc[:, 0:5]
        cols = self.X.columns
        # chuẩn hóa dữ liệu
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(self.X)

        self.X = pd.DataFrame(np_scaled, columns=cols)

    def find_cosine_similar_product(self, product_id, n=5):
        similarity = cosine_similarity(self.X)
        cosine_sim_df = pd.DataFrame(similarity, columns=self.productId, index=self.productId)
          # Lấy hàng tương ứng với product_id
        selected_row = cosine_sim_df.loc[product_id]

        # Sắp xếp các sản phẩm theo độ tương đồng và chọn n sản phẩm tương đồng nhất
        similar_products = selected_row.sort_values(ascending=False)
        similar_products = similar_products.drop(product_id)
        return similar_products.head(n)

    def find_euclid_similar_product(self, product_id, n=5):
        pairwise_dist_mat = euclidean_distances(self.X)
        euclid_sim_df = pd.DataFrame(pairwise_dist_mat,  columns=self.productId, index=self.productId)
        # Lấy hàng tương ứng với product_id
        selected_row = euclid_sim_df.loc[product_id]
        similar_products = selected_row.sort_values(ascending=True)
        similar_products = similar_products.drop(product_id)
        return similar_products.head(n)

    # Cần hỏi lại tại sao lại sử dụng cả 2 trong cùng 1 function?
    def find_similar_product(self, product_id, min_n=5):
        cosine_series = self.find_cosine_similar_product(product_id, min_n)
        euclid_series = self.find_euclid_similar_product(product_id, min_n)
        cosine_list = []
        for c, v in cosine_series.items():
            cosine_list.append(c)
        euclid_list = []
        for c, v in euclid_series.items():
            euclid_list.append(c)

        return list(set(cosine_list + euclid_list))

    def updateListProduct(self):
        # URL của API
        api_url = "https://cosmetic-ecommerce.onrender.com/api/v1/products/export-data"
        response = requests.get(api_url)
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

