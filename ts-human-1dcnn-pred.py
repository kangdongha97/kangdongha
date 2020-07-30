# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
 
# load a single file as a numpy array
def load_file(filepath):
    print(filepath)
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    print('dataframe',dataframe.values)
    return dataframe.values
 
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    print('load_group',filenames)
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    print('dataframe',data)
    loaded = dstack(loaded)
    # print('loaded',loaded)

    return loaded
 
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
    
	return X, y
 
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset-org/')
    print(trainX.shape, trainy.shape)
	# load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset-org/')
    print(testX.shape, testy.shape)

	# zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
	# one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    print('trainX',trainX)
    print('trainy', trainy)

    print('testX', testX)
    print('testy',  testy)

    return trainX, trainy, testX, testy
# plot a histogram of each variable in the dataset
def plot_variable_distributions(trainX):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	print(longX.shape)
	plt.figure()
	xaxis = None
	for i in range(longX.shape[1]):
		ax = plt.subplot(longX.shape[1], 1, i+1, sharex=xaxis)
		ax.set_xlim(-1, 1)
		if i == 0:
			xaxis = ax
		plt.hist(longX[:, i], bins=100)
	plt.show()
# fit and evaluate a model
# def evaluate_model(trainX, trainy, testX, testy):
#     verbose, epochs, batch_size = 0, 10, 32
#     n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
#     model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# fit network
#     model.summary()
#     model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# 	# evaluate model
#     _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
#     return accuracy
 
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
# load data
trainX, trainy, testX, testy = load_dataset()
# plot histograms
plot_variable_distributions(trainX)

# 	plt.show()
plt.figure()
aaa=trainX[:,0,1]    
plt.plot(aaa[0:100])
plt.plot(trainy[0:100])
plt.show()

scores = list()
repeats=1
for r in range(repeats):
    
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
    model.summary()
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
    _, score = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)


# summarize results
summarize_results(scores)
pred=model.predict(trainX[0:100])
print(pred)
 

