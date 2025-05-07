# %% Load libraries and dataset
import numpy as np
print('Libraries loaded')

date = 20250502
sampling = '5Hz'
channels = '4'
run = '1'
steps = 100

pattern = ['concentric', 'hilbert', 'honeycomb', 'rectilinear', 'triangle']

## No Feature Extraction
#X0 = np.load('4 Machine Learning' + f'/{date}_{pattern[0]}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy')
#X1 = np.load('4 Machine Learning' + f'/{date}_{pattern[1]}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy')
#X2 = np.load('4 Machine Learning' + f'/{date}_{pattern[2]}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy')
#X3 = np.load('4 Machine Learning' + f'/{date}_{pattern[3]}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy')
#X4 = np.load('4 Machine Learning' + f'/{date}_{pattern[4]}_{sampling}_{channels}_{run}_PreprocessedWith{steps}Windows.npy')

## Time Domain
#X0 = np.load('4 Machine Learning' + f'/0 {date}_{pattern[0]}_{sampling}_{channels}_{run}_TimeDomainFeatures_{steps}.npy')
#X1 = np.load('4 Machine Learning' + f'/0 {date}_{pattern[1]}_{sampling}_{channels}_{run}_TimeDomainFeatures_{steps}.npy')
#X2 = np.load('4 Machine Learning' + f'/0 {date}_{pattern[2]}_{sampling}_{channels}_{run}_TimeDomainFeatures_{steps}.npy')
#X3 = np.load('4 Machine Learning' + f'/0 {date}_{pattern[3]}_{sampling}_{channels}_{run}_TimeDomainFeatures_{steps}.npy')
#X4 = np.load('4 Machine Learning' + f'/0 {date}_{pattern[4]}_{sampling}_{channels}_{run}_TimeDomainFeatures_{steps}.npy')

## Frequency Domain
X0 = np.load('4 Machine Learning' + f'/1 {date}_{pattern[0]}_{sampling}_{channels}_{run}_FrequencyDomainFeatures_{steps}.npy')
X1 = np.load('4 Machine Learning' + f'/1 {date}_{pattern[1]}_{sampling}_{channels}_{run}_FrequencyDomainFeatures_{steps}.npy')
X2 = np.load('4 Machine Learning' + f'/1 {date}_{pattern[2]}_{sampling}_{channels}_{run}_FrequencyDomainFeatures_{steps}.npy')
X3 = np.load('4 Machine Learning' + f'/1 {date}_{pattern[3]}_{sampling}_{channels}_{run}_FrequencyDomainFeatures_{steps}.npy')
X4 = np.load('4 Machine Learning' + f'/1 {date}_{pattern[4]}_{sampling}_{channels}_{run}_FrequencyDomainFeatures_{steps}.npy')

## Wavelet Domain
family = 'mexh'
X0 = np.load('4 Machine Learning' + f'/2 {date}_{pattern[0]}_{sampling}_{channels}_{run}_WaveletFeatures_{steps}_{family}.npy')
X1 = np.load('4 Machine Learning' + f'/2 {date}_{pattern[1]}_{sampling}_{channels}_{run}_WaveletFeatures_{steps}_{family}.npy')
X2 = np.load('4 Machine Learning' + f'/2 {date}_{pattern[2]}_{sampling}_{channels}_{run}_WaveletFeatures_{steps}_{family}.npy')
X3 = np.load('4 Machine Learning' + f'/2 {date}_{pattern[3]}_{sampling}_{channels}_{run}_WaveletFeatures_{steps}_{family}.npy')
X4 = np.load('4 Machine Learning' + f'/2 {date}_{pattern[4]}_{sampling}_{channels}_{run}_WaveletFeatures_{steps}_{family}.npy')

print(X0.shape, X1.shape, X2.shape, X3.shape, X4.shape)

#%% Balance the datasets
minSamples = np.min([X0.shape[0], X1.shape[0], X2.shape[0], X3.shape[0], X4.shape[0]])
print(f'Minimum samples: {minSamples}')

X0Balanced = X0[:minSamples]
X1Balanced = X1[:minSamples]
X2Balanced = X2[:minSamples]
X3Balanced = X3[:minSamples]
X4Balanced = X4[:minSamples]

Y0Balanced = np.repeat(0, minSamples) # 0 for concentric
Y1Balanced = np.repeat(1, minSamples) # 1 for hilbert
Y2Balanced = np.repeat(2, minSamples) # 2 for honeycomb
Y3Balanced = np.repeat(3, minSamples) # 3 for rectilinear
Y4Balanced = np.repeat(4, minSamples) # 4 for triangle

print('Balanced datasets:')
print(X0Balanced.shape, X1Balanced.shape, X2Balanced.shape, X3Balanced.shape, X4Balanced.shape)
print('')
print('Labels:')
print(Y0Balanced.shape, Y1Balanced.shape, Y2Balanced.shape, Y3Balanced.shape, Y4Balanced.shape)

#%% Concatenated datasets
X = np.concatenate((X0Balanced, X1Balanced, X2Balanced, X3Balanced, X4Balanced), axis=0)
y = np.concatenate((Y0Balanced, Y1Balanced, Y2Balanced, Y3Balanced, Y4Balanced), axis=0)
print('Concatenated datasets:')
print(X.shape, y.shape)

# %% Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

print('Training:')
print(X_train.shape, y_train.shape)
print('Testing:')
print(X_test.shape, y_test.shape)

#%% Normalize the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print('Normalized training data:')
print(X_train.shape, X_test.shape)

# %% ML algorithms (Revisit Lecture Notes 7-9)
# Load libraries
from sklearn import model_selection

# covered in class
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# own experimentation
from sklearn.svm import SVC
#import tensorflow as tf

# Configure evaluation
kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'

models = [('KNN', KNeighborsClassifier(n_neighbors=5, metric='euclidean')),
         ('Decision Tree', DecisionTreeClassifier(max_depth=30, random_state=42)), 
         ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42))]

# Archived: 
# ('MLP', MLPClassifier(hidden_layer_sizes=(100, 20), max_iter=1000, activation='relu', random_state=42))
# ('SVM', SVC(kernel='rbf', C=1, gamma='scale', random_state=42))

results = []
names = []
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.3f} (±{cv_results.std():.3f})")

#%% Best model
choice = 1

clf = models[choice][1] # first index indicates the model to use
#clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

#%% Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)  # 

print(f'Training Accuracy: {round(clf.score(X_train, y_train) * 100, 2)}%') #
print(f'Testing Accuracy: {round(clf.score(X_test, y_test) * 100, 2)}%') #

print('')
print('Classification Report:\n', classification_report(y_test, y_pred))

#%% Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=pattern)

# Plot with values and colorbar
disp.plot(cmap='Blues', values_format='d')  # 'd' = integer formatting
print('Confusion Matrix:')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

#%% CHOSEN: DECISION TREE AND RANDOM FOREST

#%% PREDICTION OF ONE SAMPLE:
import pandas as pd
filePath = 'Team_1_FourChannels'

date = 20250502
pattern = 'triangle'
channels = '4'
sampling = '5Hz'
run = '1'
steps = 100

# p - print bed accelerometer; n - nozzle; s - sound sensor (Right, Left)
df = pd.read_csv(filePath + f'/{date}_{pattern}_{sampling}_{channels}_{run}.csv', 
                 names=['Time', 'Yp', 'Yn', 'SoundR', 'SoundL'])
dfNew = df['Time'].str.split(',', expand=True)
dfNew = dfNew.apply(pd.to_numeric, errors='coerce')

i = 300 
j = i+steps
dfNew = dfNew[i:j]

time = dfNew[0]  # time in seconds
yP, yN = np.array(dfNew[1]), np.array(dfNew[2])
sR, sL = np.array(dfNew[3]), np.array(dfNew[4])

## Wavelet Domain
import scipy.stats as stats
import pywt

sp = np.diff(time).mean()
family = 'mexh'
scales = np.arange(1, 500) # Perplexity: np.geomspace(1, 1024, num=75)

printBedNew = pywt.cwt(yP, scales, family, sampling_period=sp)[0]
nozzleNew = pywt.cwt(yN, scales, family, sampling_period=sp)[0]
soundRightNew = pywt.cwt(sR, scales, family, sampling_period=sp)[0]
soundLeftNew = pywt.cwt(sL, scales, family, sampling_period=sp)[0]

printBedNew = printBedNew.reshape(len(printBedNew), -1) 
nozzleNew = nozzleNew.reshape(len(nozzleNew), -1)
soundRightNew = soundRightNew.reshape(len(soundRightNew), -1)
soundLeftNew = soundLeftNew.reshape(len(soundLeftNew), -1)

printBedFDF, nozzleFDF = [], []
soundRightFDF, soundLeftFDF = [], []

for i in printBedNew:
    printBedFDF.append([np.max(i), np.sum(i), np.mean(i), np.var(i), 
                        np.max(np.abs(i)), stats.skew(i), stats.kurtosis(i)])
for j in nozzleNew:
    nozzleFDF.append([np.max(j), np.sum(j), np.mean(j), np.var(j), 
                      np.max(np.abs(j)), stats.skew(j), stats.kurtosis(j)])
for k in soundRightNew:
    soundRightFDF.append([np.max(k), np.sum(k), np.mean(k), np.var(k), 
                          np.max(np.abs(k)), stats.skew(k), stats.kurtosis(k)])
for l in soundLeftNew:
    soundLeftFDF.append([np.max(l), np.sum(l), np.mean(l), np.var(l), 
                         np.max(np.abs(l)), stats.skew(l), stats.kurtosis(l)])

printBedFDF, nozzleFDF = np.array(printBedFDF), np.array(nozzleFDF)
soundRightFDF, soundLeftFDF = np.array(soundRightFDF), np.array(soundLeftFDF)
wdf = np.concatenate([printBedFDF, nozzleFDF, soundRightFDF, soundLeftFDF])

print('Prediction:', np.mean(clf.predict(wdf)))
# %%
