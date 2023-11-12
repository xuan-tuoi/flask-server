# -- coding: utf-8
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn import preprocessing
import pandas as pd


class RecommenderModel:
    def __init__(self, features_csv_path):
        # Read data
        data = pd.read_csv(features_csv_path, index_col='filename')

        # Extract labels
        self.labels = data[['label']]

        # Drop labels from original dataframe
        data = data.drop(columns=['length', 'label'])

        # Scale the data
        self.data_scaled = preprocessing.scale(data)

    def find_cosine_similar_songs(self, name, n=5):
        similarity = cosine_similarity(self.data_scaled)

        # Convert into a dataframe and then set the row index and column names as labels
        sim_df_labels = pd.DataFrame(similarity)
        sim_df_names = sim_df_labels.set_index(self.labels.index)
        sim_df_names.columns = self.labels.index

        # Find songs most similar to another song
        series = sim_df_names[name].sort_values(ascending=False)

        # Remove cosine similarity == 1 (songs will always have the best match with themselves)
        series = series.drop(name)
        for i, v in series.head(n).items():
            print('name: ', i, 'cosine similar: ', v)

        # Display the 5 top matches
        return series.head(n)

    def find_euclid_similar_songs(self, name, n=5):
        pairwise_dist_mat = euclidean_distances(self.data_scaled)

        # Convert into a dataframe and then set the row index and column names as labels
        sim_df_labels = pd.DataFrame(pairwise_dist_mat)
        sim_df_names = sim_df_labels.set_index(self.labels.index)
        sim_df_names.columns = self.labels.index

        # Find songs most similar to another song
        series = sim_df_names[name].sort_values(ascending=True)

        # Remove cosine similarity == 1 (songs will always have the best match with themselves)
        series = series.drop(name)
        for i, v in series.head(n).items():
            print('name: ', i, 'euclid distance: ', v)

        # Display the 5 top matches
        return series.head(n)

    def find_similar_songs(self, name, min_n=5):
        cosine_series = self.find_cosine_similar_songs(name, min_n)
        euclid_series = self.find_euclid_similar_songs(name, min_n)
        cosine_list = []
        for c, v in cosine_series.items():
            cosine_list.append(c)
        euclid_list = []
        for c, v in euclid_series.items():
            euclid_list.append(c)
        return list(set(cosine_list + euclid_list))
