import tensorflow as tf
from tensorflow.keras import layers
from google.colab import drive

drive.mount('/content/drive')

# Function to resize images to 512x512
def resize_image(image):
    image = tf.image.resize(image, [128, 128], method='bilinear')
    return image

def quantize_weights(model):
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
            weights = layer.get_weights()
            quantized_weights = [np.round(w * 127) / 127 for w in weights]  # INT8 quantization
            layer.set_weights(quantized_weights)

train_dir = "/content/drive/MyDrive/CSC480/Candy/datasets/train_set_100"#"./train_set_100"
test_dir = "/content/drive/MyDrive/CSC480/Candy/datasets/test_set_100"#"./test_set_100"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(720, 1280),
    color_mode="rgb",
    batch_size=10,
    label_mode='int'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(720, 1280),
    color_mode="rgb",
    batch_size=5,
    label_mode='int'
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
])

normalization_layer = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

train_dataset = train_dataset.map(lambda x, y: (tf.image.resize(x, [128, 128]), y))
test_dataset = test_dataset.map(lambda x, y: (tf.image.resize(x, [128, 128]), y))

# class_weights = {0: 1.0, 1: 1.2857, 2: 1.125}
class_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

# Update model architecture for better performance
model = tf.keras.models.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Fewer filters
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),  # Fewer filters
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),  # Fewer filters
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),  # Fewer neurons
    layers.Dense(16, activation='relu'),  # Fewer neurons
    layers.Dense(4, activation='softmax')
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Smaller learning rate
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(train_dataset, steps_per_epoch=10, epochs=20, class_weight=class_weights, validation_data=test_dataset, callbacks=[early_stopping])

# Evaluate the model on the test dataset
model.evaluate(test_dataset, verbose=2)

# Save the trained model
quantize_weights(model)
model.save('quantized_candy_classifier_model.h5')

# Convert the model to TensorFlow Lite format
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# Save the TFLite model to a file
# with open('simplified_candy_classifier_model.tflite', 'wb') as f:
#     f.write(tflite_model)