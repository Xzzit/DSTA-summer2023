import csv
import numpy as np
import pickle
import os

csv_dir = '/mnt/e/project/summer2023/dsta-train-v2.csv'
root = 'Data'

with open(csv_dir) as f:
    next(f)
    reader = csv.reader(f)

    skeletons_all_train = []
    labels_all_train = []

    group_id = 0
    xy_frame_data = np.empty((2, 21, 0), float)
    for row in reader:
        
        if group_id == int(row[0]):
            xy_data = np.empty((2, 0), float)
            for i in row[2:]:
                x, y = map(float, i.split(','))
                xy_data = np.append(xy_data, np.expand_dims([x, y], axis=1), axis=1)  # 2 x 21
            xy_frame_data = np.append(xy_frame_data, np.expand_dims(xy_data, axis=2), axis=2)  # 2 x 21 x Frames

        else:
            labels_all_train.append(action_id)
            group_id = int(row[0])
            skeletons_all_train.append(np.expand_dims(xy_frame_data.transpose((0, 2, 1)), axis=3))
            xy_frame_data = np.empty((2, 21, 0), float)
            xy_data = np.empty((2, 0), float)
            for i in row[2:]:
                x, y = map(float, i.split(','))
                xy_data = np.append(xy_data, np.expand_dims([x, y], axis=1), axis=1)  # 2 x 21
            xy_frame_data = np.append(xy_frame_data, np.expand_dims(xy_data, axis=2), axis=2)  # 2 x 21 x Frames
        
        action_id = int(row[1])-1

pickle.dump(skeletons_all_train, open(os.path.join(root, 'summer_train_skeleton.pkl'), 'wb'))
pickle.dump(labels_all_train, open(os.path.join(root, 'summer_train_label.pkl'), 'wb'))