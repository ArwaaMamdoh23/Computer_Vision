
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout


train_dir = r"D:\GitHub\Computer_Vision\Teeth_Dataset\Training"
test_dir = r"D:\GitHub\Computer_Vision\Teeth_Dataset\Testing"
val_dir = r"D:\GitHub\Computer_Vision\Teeth_Dataset\Validation"


img_size = (256, 256)
batch_size = 32
datagen = ImageDataGenerator(
    rescale=1.0/255,  
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2,  
    horizontal_flip=True  
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

print("Data generators reloaded successfully.")

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout to reduce overfitting
    Dense(7, activation='softmax', kernel_regularizer=regularizers.l2(0.01)) 
])


model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

base_model.trainable = True
for layer in base_model.layers[:143]:  # Unfreeze only the last few layers
    layer.trainable = False


model.compile(optimizer=Adam(learning_rate=0.00001),  
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,  
    verbose=1
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# Calculate Precision and Recall
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=-1)
y_true = test_generator.classes

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
