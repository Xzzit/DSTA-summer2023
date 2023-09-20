import pickle

file1 = open('Data/summer_train_skeleton.pkl', 'rb')
data = pickle.load(file1)

file2 = open('Data/summer_train_label.pkl', 'rb')
label = pickle.load(file2)

for i, j in zip(data, label):
    print(i.shape, j)