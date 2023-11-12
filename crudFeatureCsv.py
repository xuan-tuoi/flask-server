# -- coding: utf-8
import csv
import pandas as pd


def add_song_features_to_data_csv(filepath, pandas_data, label):
    filename = filepath.split('/')[-1]
    features_data = 'csvData/data.csv'

    fields = ['filename', 'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
              'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
              'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean',
              'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean',
              'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
              'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean',
              'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
              'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean',
              'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var',
              'mfcc20_mean', 'mfcc20_var', 'label']

    with open(features_data, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        for row in reader:
            if row['filename'] == str(filename):
                # print('found! don't need to add new row to data.csv')
                return
    # print('not found! adding new row to data.csv')
    to_append = f'{filename} 661794'
    for p in pandas_data[0]:
        to_append += f' {p}'
    to_append += f' {label}'
    # print(to_append)
    file = open(features_data, 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


def remove_song_features_from_data_csv(filepath):
    filename = filepath.split('/')[-1]
    features_data = 'csvData/data.csv'

    df = pd.read_csv(features_data)
    df = df.drop(df[df.filename == filename].index)
    df.to_csv(features_data, index=False)
