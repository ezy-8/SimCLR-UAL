#%% Experiment setup
dataset = 'CIFAR10'
modelName = '3DCNN'
amount = '100P'
trial = 10
epochs = 100
batchSize = 256

activeSamples = 1000 #
print('Ready for experiment')

#%% Load dataset and reorganize (CIFAR10)
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Split CIFAR10 into 10% and split this into train/test
(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()

subset = 1 # 30, 60, 100 percent

smallerTrain, smallerTest = int(len(xTrain) * subset), int(len(xTest) * subset)

xTrain, xTest = xTrain[0:smallerTrain], xTest[0:smallerTest]

yTrain, yTest = yTrain[0:smallerTrain], yTest[0:smallerTest]

print(xTrain.shape, xTest.shape)
print(yTrain.shape, yTest.shape)

#%% Data preparation for TensorFlow
from keras import layers, models

xTrain, xTest = tf.convert_to_tensor(xTrain) / 255, tf.convert_to_tensor(xTest) / 255
yTrain, yTest = yTrain.flatten(), yTest.flatten()

print(xTrain.shape, xTest.shape)
print(yTrain.shape, yTest.shape)


#%% Training
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

# Full-pretrained model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), 
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%% Active Learning Part 1: fit
model.fit(xTrain, yTrain, epochs=100, batch_size=batchSize)
predictions = model.predict(xTrain)

#%% Active Learning Part 2: represent in 2D space
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

embed = TSNE(n_components=2, random_state=42).fit_transform(predictions)

plt.figure(0)
plt.scatter(x=embed[:, 0], y=embed[:, 1])
plt.title("t-SNE or UMAP Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#%% Active Learning Part 3: K-means clustering
from sklearn.cluster import KMeans
clustering = KMeans(clusters=10, random_state=42).fit(embed)
centers = clustering.cluster_centers_
labels = clustering.labels_
print('Finish assigning clusters')

plt.figure(1)
plt.title(f'K-Means clusters in TSNE space using three-layer DCNN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

s = plt.scatter(embed[:, 0], embed[:, 1], c=labels)
plt.colorbar(s, label='Cluster Label')

c = plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='Centroids')
plt.legend()
plt.show()

#%% Active Learning Part 4: Choose samples near the center
import numpy as np
from sklearn.cluster import KMeans

clusters = 10  # Total clusters
closestPoints = []

for clusterID in range(clusters):
    # Get points in current cluster
    clusterMask = (labels == clusterID)
    clusterPoints = embed[clusterMask]
    
    # Calculate distances to centroid
    centroid = centers[clusterID]
    distances = np.linalg.norm(clusterPoints - centroid, axis=1)
    
    # Get indices of two closest points
    closestIndices = np.argsort(distances)[:activeSamples] # buyer beware
    
    # Store results
    closestPoints.append(closestIndices)

closestPoints = np.array(closestPoints)

#%% Obtain subset (informative samples based on closest points to the kmeans center)
xTrainSub = []
yTrainSub = []

for i in closestPoints:
    for j in i:
        xTrainSub.append(xTrain[j])
        yTrainSub.append(yTrain[j])

xTrainSub = np.array(xTrainSub)
yTrainSub = np.array(yTrainSub)
print(xTrainSub.shape, yTrainSub.shape)

#%% Training
print("Starting Classification Training \n")
import time
start = time.time()

history = model.fit(xTrainSub, yTrainSub, validation_split=0.2, epochs=epochs, batch_size=batchSize)

end = time.time()
totalTrainingTime = end - start
print(f"\nTotal training time: {totalTrainingTime:.2f} seconds")

#%% Testing
import numpy as np
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

testLoss, testAccuracy = model.evaluate(xTest, yTest)
print(f"Test Accuracy: {100 * testAccuracy:.4f} %")

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

truth = yTest

predictions = model.predict(xTest)
predicted = np.argmax(predictions, axis=1)

trainingAccuracy = history.history['accuracy'][-1] * 100
testingAccuracy = testAccuracy * 100

# Classification Report
print("\nClassification Report:")
print(classification_report(truth, predicted))

# Plot confusion matrix
cm = confusion_matrix(truth, predicted)
results = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
results.plot(cmap='Blues', colorbar=True)
plt.title("Confusion Matrix")
plt.savefig(f'/UAL{trial}_{modelName}_{amount}_TrainTime{round(totalTrainingTime,2)}_TrainAcc{round(trainingAccuracy,2)}_TestAcc{round(testingAccuracy,2)}_Epochs{epochs}.png')
plt.show()

# %%
