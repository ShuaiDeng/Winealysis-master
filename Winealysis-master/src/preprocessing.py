#Used to create Train, Test and Validation slices of the dataset.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

#import dataset
wine_data_raw_pd = pd.read_csv('../data/_raw/winemag-data-130k-v2.csv')

#export as pandaframe and reload
wine_data_raw_pd.to_csv('../data/winemag-data-130k-v2.csv', index=False, quoting=1)
wine_data_raw_pd = pd.read_csv('../data/winemag-data-130k-v2.csv')

#-----------split dataset-------------
#1. training dataset: 80%
#2. test dataset: 10%
#3. calidation dataset: 10%

#Using scikit-learn we split the dataset into training and test+validation
training_data_raw_pd, test_data_raw_pd = train_test_split(wine_data_raw_pd, shuffle=True, train_size=0.8, test_size=0.2)

#We further split test+validation into seperate test & validation datasets.
test_data_raw_pd, validation_data_raw_pd = train_test_split(test_data_raw_pd, shuffle=True, train_size=0.5, test_size=0.5)

#export datasets
training_data_raw_pd.to_csv('../data/training_data_raw.csv', index=False, quoting=1)
test_data_raw_pd.to_csv('../data/test_data_raw.csv', index=False, quoting=1)
validation_data_raw_pd.to_csv('../data/validation_data_raw.csv', index=False, quoting=1)
