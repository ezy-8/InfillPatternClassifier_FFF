# %% Load libraries and dataset
import numpy as np
print('Libraries loaded')

date = 20250430
sampling = '1Hz6Ch'
run = '1'
steps = 2

pattern = ['concentric', 'hilbert', 'honeycomb', 'rectilinear', 'triangle']

X0 = np.load('4 Machine Learning' + f'/{date}_{pattern[0]}_{sampling}_{run}_PreprocessedWith{steps}Windows.npy')
X1 = np.load('4 Machine Learning' + f'/{date}_{pattern[1]}_{sampling}_{run}_PreprocessedWith{steps}Windows.npy')
X2 = np.load('4 Machine Learning' + f'/{date}_{pattern[2]}_{sampling}_{run}_PreprocessedWith{steps}Windows.npy')
X3 = np.load('4 Machine Learning' + f'/{date}_{pattern[3]}_{sampling}_{run}_PreprocessedWith{steps}Windows.npy')
X4 = np.load('4 Machine Learning' + f'/{date}_{pattern[4]}_{sampling}_{run}_PreprocessedWith{steps}Windows.npy')

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

#%% Apply PCA
'''from sklearn.decomposition import PCA
pca = PCA(n_components=5, random_state=42)  # Keep 95% of variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print('PCA applied:')
print(X_train.shape, X_test.shape)'''

# %% Compare Algorithms
# Load libraries
from sklearn import model_selection, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Configure evaluation
kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'

models = [
    ('LR', LogisticRegression(max_iter=5000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC()),
    ('RF', RandomForestClassifier()),
    ('Ada', AdaBoostClassifier())
]

results = []
names = []

for name, model in models:
    cv_results = model_selection.cross_val_score(
        model, X_train, y_train, cv=kfold, scoring=scoring
    )
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.3f} (±{cv_results.std():.3f})")

#%% Best model
choice = 6 # 0: LR, 1: LDA, 2: KNN, 3: CART, 4: NB, 5: SVM, 6: RF, 7: Ada

clf = models[choice][1] # first index indicates the model to use
clf = SVC(kernel='rbf')  # SVM with RBF kernel
clf.fit(X_train, y_train)
print(f'Best model: {models[choice][0]}')

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5, 5), activation='tanh', solver='adam', max_iter=500, random_state=1)
mlp.fit(X_train, y_train)

#%% Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)  # 
print(f'Accuracy: {round(clf.score(X_test, y_test) * 100, 2)}%') #
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

# 04/30/2025 - Highest test accuracy: 63.16% with SVC (linear kernel) and PCA
# %%
