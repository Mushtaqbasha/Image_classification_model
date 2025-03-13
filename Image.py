import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    # Convolutional layer with 32 filters, kernel size 3x3, activation function 'relu'
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),  # MaxPooling layer
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten the 3D matrix to 1D
    layers.Flatten(),
    
    # Fully connected layer with 64 neurons
    layers.Dense(64, activation='relu'),
    
    # Output layer with 10 classes (for CIFAR-10)
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=30, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Save the model in TensorFlow's SavedModel format with versioning
model.save('cifar10_model_v1.keras')

# Load the model
try:
    loaded_model = tf.keras.models.load_model('cifar10_model_v1.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Make a prediction
try:
    predictions = loaded_model.predict(test_images)
except Exception as e:
    print(f"Error making predictions: {e}")
    exit()

# Show the first image and its predicted class with a title
plt.imshow(test_images[0])
plt.title(f'Predicted: {predictions[0].argmax()}')
plt.show()

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
