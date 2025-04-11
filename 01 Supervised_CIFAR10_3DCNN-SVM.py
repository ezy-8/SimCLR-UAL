#%% Experiment setup
dataset = 'CIFAR10'
modelName = '3DCNN-SVM'
methodName = 'Supervised'
trial = 10
epochs = 100

batchSize = 256

amount = '100P'
subset = 1 # 30, 60, 100 percent

print('Ready for experiment')

#%% Load dataset and reorganize (CIFAR)
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Split CIFAR10 into 10% and split this into train/test
(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()

smallerTrain, smallerTest = int(len(xTrain) * subset), int(len(xTest) * subset)

xTrain, xTest = xTrain[0:smallerTrain], xTest[0:smallerTest]

yTrain, yTest = yTrain[0:smallerTrain], yTest[0:smallerTest]

print(xTrain.shape, xTest.shape)
print(yTrain.shape, yTest.shape)

# Classes - ordered 0 to 9: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

#%% Data preparation for TensorFlow
from keras import layers, models

xTrain, xTest = tf.convert_to_tensor(xTrain) / 255, tf.convert_to_tensor(xTest) / 255
yTrain, yTest = yTrain.flatten(), yTest.flatten()

print(xTrain.shape, xTest.shape)
print(yTrain.shape, yTest.shape)

#%% Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(batchSize, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print('Loaded classifier')

#%% Training
print("Starting Classification Training \n")

import matplotlib.pyplot as plt
import time

start = time.time()

history = model.fit(xTrain, yTrain, validation_split=0.2, epochs=epochs, batch_size=batchSize)

end = time.time()

totalTrainingTime = end - start
print(f"\nTotal training time: {totalTrainingTime:.2f} seconds")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% Feature Extraction
extraction = models.Model(inputs=model.input, outputs=model.layers[-1].output)

#%% Testing with SVM classifier and GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

trainFeatures = extraction.predict(xTrain)
testFeatures = extraction.predict(xTest)

hyperParameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 1], 'kernel': ['rbf', 'linear']}

svm = GridSearchCV(SVC(), hyperParameters, refit=True, verbose=2, cv=5)
svm.fit(trainFeatures, yTrain)

print('Best parameters:', svm.best_params_)
print('\nBest score:', svm.best_score_)

#%% Evaluate model on test
bestModel = svm.best_estimator_
test_accuracy = bestModel.score(testFeatures, yTest)
print(f"Test Accuracy: {100 * test_accuracy:.4f} %")

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

truth = yTest

predictions = bestModel.predict(testFeatures)

trainingAccuracy = svm.best_score_ * 100
testingAccuracy = test_accuracy * 100

# Classification Report
print("\nClassification Report:")
print(classification_report(truth, predictions))

# Plot confusion matrix
cm = confusion_matrix(truth, predictions)
results = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
results.plot(cmap='Blues', colorbar=True)
plt.title("Confusion Matrix")
plt.savefig(f'/V{trial}_{modelName}_{methodName}_{amount}_PreTrainTime0_TrainTime{round(totalTrainingTime,2)}_TrainAcc{round(trainingAccuracy,2)}_TestAcc{round(testingAccuracy,2)}_Epochs{epochs}.png')
plt.show()

# %%
