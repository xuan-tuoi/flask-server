import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn import preprocessing

# Đọc dữ liệu từ file CSV
data = pd.read_csv('csvData/product.csv')

# =========== B1: Preprocessing ========

# Loại bỏ cột 'id' nếu có
if 'id' in data.columns:
    data = data.drop('id', axis=1)

    # Loại bỏ cột 'product_name' nếu có
if 'product_name' in data.columns:
    data = data.drop('product_name', axis=1)

shop_mapping = {'40c9cb9c-1628-4995-b8a6-7a5ce183e087': 1, '8c4451cf-3617-4d71-b003-c09d2eb5aa53': 2, '9a420be2-4df3-4595-a500-2211de5a9701': 3}  # Ánh xạ giá trị chuỗi thành số
data.user_id = data.user_id.map(shop_mapping)


categories = ['Eye cream', 'Exfoliator', 'Peptide', 'Cleansing oil', 'Sun cream', 'Cleanser', 'Cream', 'Face Mask', 'Sunscreen', 'Bath Salts', 'Body Wash', 'HA',
              'Toner', 'Mask', 'Mist', 'Body Lotion', 'Gel', 'Moisturizer', 'Hair', 'Serum']

# Tạo một đối tượng LabelEncoder và áp dụng cho cột 'category'
label_encoder = LabelEncoder()
# Áp dụng LabelEncoder để chuyển đổi danh mục thành số
categories_encoded = label_encoder.fit_transform(categories)
data['product_category'] = label_encoder.fit_transform(data['product_category'])

data = data.dropna() # loại bỏ giá trị null
data = data.drop_duplicates() # loại bỏ giá trị bị trùng lặp

X = data.iloc[:, 0:5]

# Lưu danh sách tên cột
column_names = data.columns

# chuẩn hóa dữ liệu
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)

df = pd.DataFrame(np_scaled, columns=column_names)

print(df.head())

# # ===========B2: cosine  ===========
# similarity = cosine_similarity(df)
#
# # Tạo DataFrame từ ma trận cosine similarity
# cosine_sim_df = pd.DataFrame(similarity, columns=df.index, index=df.index)
#
# # In ra ma trận cosine similarity
# print(cosine_sim_df)


# =================== B3: sử dụng Euclid ==========
# euclid = euclidean_distances(df)
#
# euclid_df = pd.DataFrame(euclid, columns=df.index, index=df.index)
# print(euclid_df)
