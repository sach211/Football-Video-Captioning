# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import os
import numpy as np
import pandas as pd

labels = pd.read_csv('/home/sachi/Documents/football/labels.csv')



"""cv2.imshow('image', img)
k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    
for i in range(576):
    for j in range(720):
        if img[i, j, 0] + 15 >= img[i, j, 1] or img[i, j, 2] + 15 >= img[i, j, 1]:
            img[i, j] = [0, 0, 0]"""
                     


green = []
ad = []
nongreen = []
label = []

l = os.listdir('/home/sachi/Documents/football/SEV_GET/All')
sort_l = [0] * 4100
for image in l:
    ind = image.split('.')[0]
    ind = int(ind)
    sort_l[ind - 1]= image
    

for image in sort_l:
    img = cv2.imread(os.path.join('/home/sachi/Documents/football/SEV_GET/All', image))
    row, column = img.shape[:2]
    countn, countg, counta = 0,0,0
    for i in range(row):
        for j in range(column):
            if img[i, j, 0] + 15 >= img[i, j, 1] or img[i, j,2] +15 >= img[i, j, 1]:
                countn = countn + 1
            elif img[i, j, 2] + 50 >= img[i, j, 1] or img[i, j,0] + 50 >= img[i, j, 1]:
                countg = countg + 1
            else:
                counta = counta + 1
    green.append(countg)
    ad.append(counta)
    nongreen.append(countn)    
    
label =  labels['Annotations']
    
fin_label = pd.DataFrame({'green': green, 'ad': ad, 'non_green': nongreen, 'label': label}, columns = ['green', 'ad', 'non_green', 'label'])
fin_label.to_csv('/home/sachi/Documents/football/SEV_GET/zoom.csv')

"""
import seaborn
import seaborn as sb
import pandas as pd
x = fin_label[['green', 'ad', 'non_green', 'label']].groupby('label').mean().reset_index()
plot_df = pd.melt(x, id_vars = 'label', var_name = "dist", value_name = "mean")
sb.factorplot(x = 'label', y = 'mean', hue = 'dist', data = plot_df, kind = 'bar')
"""
non_field = fin_label[fin_label['label'] == 0]
non_field = non_field.append(fin_label[fin_label['label'] == 4])

field = fin_label[fin_label['label'] == 1]
field = field.append(fin_label[fin_label['label'] == 2])

pred_list = []
for i in list(non_field.index.values):
    if non_field.loc[i]['non_green'] < non_field.loc[i]['ad']: 
        pred_list.append(4)
    else:
        pred_list.append(0)
        
count = 0
k = 0;
for i in list(non_field.index.values):
    if non_field.loc[i]['label'] != pred_list[k]:
        count = count + 1
    k = k + 1
    
print 100.0 * (1.0 - (float(count)/float(len(non_field))))
"""

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

input_data = fin_label[['green', 'ad', 'non_green']].values
labels = fin_label[['label']].values.ravel()

(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(input_data, labels, train_size=0.75, random_state=1) 
prediction_model = RandomForestClassifier()
cv_prediction_model = RandomForestClassifier()

prediction_model.fit(training_inputs, training_classes)
score = prediction_model.score(testing_inputs, testing_classes)

print "Normal score:"
print score   

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
cv_scores = []

p = {'max_features':[1,2,3]}
c_v = StratifiedKFold(labels, n_folds = 10)

grid_search = GridSearchCV(cv_prediction_model, param_grid = p, cv = c_v)
grid_search.fit(input_data,labels)
cv_prediction_model = grid_search.best_estimator_
print(grid_search.best_params_)
print(grid_search.best_score_)
"""