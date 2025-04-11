#%% Experiment setup
dataset = 'CIFAR10'
modelName = '3DCNN'
methodName = 'SimCLR'
trial = 10
epochs = 100

batchSize = 256

amount = '100P'
subset = 1 # 30, 60, 100 percent

print('Ready for experiment')

#%% Load dataset and reorganize (CIFAR10)
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Split CIFAR10 into 10% and split this into train/test
(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()

smallerTrain, smallerTest = int(len(xTrain) * subset), int(len(xTest) * subset)

xTrain, xTest = xTrain[0:smallerTrain], xTest[0:smallerTest]

yTrain, yTest = yTrain[0:smallerTrain], yTest[0:smallerTest]

print(xTrain.shape, xTest.shape)
print(yTrain.shape, yTest.shape)

# Classes: ordered 0 to 9
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

#%% Data preparation for TensorFlow
from keras import layers, models

xTrain, xTest = tf.convert_to_tensor(xTrain) / 255, tf.convert_to_tensor(xTest) / 255
yTrain, yTest = yTrain.flatten(), yTest.flatten()

print(xTrain.shape, xTest.shape)
print(yTrain.shape, yTest.shape)

#%% Load base encoder
def customEncoder():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])

print('Loaded base encoder')

#%% Define the SimCLR model
class SimCLR(tf.keras.Model):
    def __init__(self, base_encoder, projection_dim):
        super(SimCLR, self).__init__()
        self.base_encoder = base_encoder
        self.projection_head = tf.keras.Sequential([
            layers.Dense(projection_dim, activation='relu')
        ])

    def call(self, x):
        # Forward pass through the base encoder
        features = self.base_encoder(x)
        projections = self.projection_head(features)
        return projections

# NT-Xent Loss (Contrastive Loss)
def nt_xent_loss(z_i, z_j, temperature=0.2):
    """Compute normalized temperature-scaled cross-entropy loss."""
    z_i = tf.math.l2_normalize(z_i, axis=1)  # Normalize embeddings
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # Concatenate positive pairs and compute similarity matrix
    representations = tf.concat([z_i, z_j], axis=0)
    similarity_matrix = tf.matmul(representations, representations, transpose_b=True)

    # Create positive and negative masks
    batch_size = tf.shape(z_i)[0]
    labels = tf.one_hot(tf.range(batch_size), 2 * batch_size)
    labels = tf.concat([labels, labels], axis=0)

    # Scale similarity scores by temperature
    logits = similarity_matrix / temperature

    # Compute the loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

print('Loaded SimCLR Implementation, and NT-Xent Loss')

#%% Data Augmentation for Contrastive Learning

def dataAugmentation(input):
    return tf.keras.Sequential([
    
    # Center Crop and Resizing
    layers.CenterCrop(16, 16),
    layers.Resizing(32, 32, interpolation='bicubic'),

    # Random Color Distortion
    layers.RandomBrightness(1),
    layers.RandomContrast(1),

    # Normalize
    layers.Rescaling(1.0 / 255.0)

])

print('Loaded Data Augmentation')

#%% Pre-Training
inputShape = (32, 32, 3)

# Instantiate the custom encoder and SimCLR model
custom_encoder = customEncoder()
simclr_model = SimCLR(base_encoder=custom_encoder, projection_dim=batchSize)

# Create data augmentations for two views of each image
augmentor = dataAugmentation(inputShape)

def preprocess(image):
    return augmentor(image), augmentor(image)

# Load CIFAR-10 dataset
dataset = (
    tf.data.Dataset.from_tensor_slices(xTrain)
    .map(preprocess)
    .batch(batchSize)
    .prefetch(tf.data.AUTOTUNE)
)

import time
start = time.time()

fewShots = 10

# Training loop with different optimizer
for epoch in range(fewShots):
    print(f"Pre-training Epoch {epoch + 1}/{fewShots}")
    
    for step, (x_batch_1, x_batch_2) in enumerate(dataset):
        with tf.GradientTape() as tape:
            z_i = simclr_model(x_batch_1)  # Projection for first view
            z_j = simclr_model(x_batch_2)  # Projection for second view
            
            ntxloss = nt_xent_loss(z_i, z_j)

        gradients = tape.gradient(ntxloss, simclr_model.trainable_variables)
        tf.optimizers.SGD().apply_gradients(zip(gradients, simclr_model.trainable_variables))
        
        
        print(f"Step {step}: Loss: {ntxloss.numpy()}")

end = time.time()
ptraintime = end - start
print(f"\nTotal pre-training time: {ptraintime:.2f} seconds")

#%% Pre-training setup
pretrainedEncoder = simclr_model.base_encoder

pretrainedEncoder.trainable = False

model = models.Sequential([
    pretrainedEncoder, # Feature backbone network 

    simclr_model.projection_head, # Projection head

    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Full-pretrained model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%% Training
print("Starting Classification Training \n")

start = time.time()

history = model.fit(xTrain, yTrain, validation_split=0.2,
                    epochs=epochs, batch_size=batchSize)

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

test_loss, test_accuracy = model.evaluate(xTest, yTest)
print(f"Test Accuracy: {100 * test_accuracy:.4f} %")

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

truth = yTest

predictions = model.predict(xTest)
predicted = np.argmax(predictions, axis=1)

trainingAccuracy = history.history['accuracy'][-1] * 100
testingAccuracy = test_accuracy * 100

# Classification Report
print("\nClassification Report:")
print(classification_report(truth, predicted))

# Plot confusion matrix
cm = confusion_matrix(truth, predicted)
results = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
results.plot(cmap='Blues', colorbar=True)
plt.title("Confusion Matrix")
plt.savefig(f'/V{trial}_{modelName}_{methodName}_{amount}_PreTrainTime{round(ptraintime,2)}_TrainTime{round(totalTrainingTime,2)}_TrainAcc{round(trainingAccuracy,2)}_TestAcc{round(testingAccuracy,2)}_Epochs{epochs}.png')
plt.show()

# %%
