from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


model = Sequential([
    # Convolutional Block 1
    Conv2D(32, kernel_size=3, activation='relu', padding='same',
           input_shape=(32, 32, 3), name='conv1'),
    MaxPooling2D(pool_size=2, name='pool1'),

    # Convolutional Block 2
    Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv2'),
    MaxPooling2D(pool_size=2, name='pool2'),

    # Convolutional Block 3
    Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv3'),
    MaxPooling2D(pool_size=2, name='pool3'),

    # Fully Connected Layers
    Flatten(name='flatten'),
    Dense(128, activation='relu', name='fc1'),
    Dropout(0.5, name='dropout'),
    Dense(10, activation='softmax', name='output')
], name='CIFAR10_CNN')

# Save the model architecture to a file
model.save('cifar10_cnn_architecture.keras')

# Print the model summary to visualize the architecture
model.summary()
