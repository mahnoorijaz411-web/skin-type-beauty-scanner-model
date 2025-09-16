from google.colab import files
uploaded = files.upload()

import os
print(os.listdir("/content"))

import zipfile, os

zip_path = "/content/archive(1).zip"   
extract_path = "/content/dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted to:", extract_path)
print("Folders inside dataset:", os.listdir(extract_path))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

base_dir = "/content/dataset/Oily-Dry-Skin-Types"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

# Image size and batch
img_size = (150, 150)
batch_size = 32

# Data Generators with Augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir,
                                              target_size=img_size,
                                              batch_size=batch_size,
                                              class_mode='categorical')

val_gen = val_datagen.flow_from_directory(val_dir,
                                          target_size=img_size,
                                          batch_size=batch_size,
                                          class_mode='categorical')

test_gen = test_datagen.flow_from_directory(test_dir,
                                            target_size=img_size,
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(train_gen,
                    epochs=30,
                    validation_data=val_gen)

# Save model
model.save("skin_type_model.h5")

from google.colab import files
uploaded = files.upload ()   #upload image

import os
print(list(uploaded.keys()))

import os

base_dir = "/content/dataset/Oily-Dry-Skin-Types/train"
for cls in ["oily", "dry", "normal"]:
    print(cls, ":", len(os.listdir(os.path.join(base_dir, cls))))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze pre-trained layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax")  # oily, dry, normal
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "/content/dataset/Oily-Dry-Skin-Types/train"
valid_dir = "/content/dataset/Oily-Dry-Skin-Types/valid"

# Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),   # MobileNet prefers 224x224
    batch_size=32,
    class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Base Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False   # transfer learning ke liye

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(3, activation="softmax")(x)  # 3 classes: oily, dry, normal

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=15,
    callbacks=[early_stop]
)

# Step 3: Fine-tuning
base_model.trainable = True   # base model layers trainable 

# Lekin sirf last 50 layers ko hi train karte hain (not overfit)
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Re-compile model with lower learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Continue training
fine_tune_epochs = 10
total_epochs = 15 + fine_tune_epochs

history_fine = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stop]
)

import matplotlib.pyplot as plt

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training & Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training & Validation Loss")
plt.show()

from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# remedies dictionary
remedies = {
    "oily": {
        "Skincare": "- Gel-based cleanser, oil-free moisturizer, non-comedogenic sunscreen",
        "Home Remedies": "- Multani mitti (Fuller's Earth) mask\n- Aloe vera gel\n- Avoid heavy creams"
    },
    "dry": {
        "Skincare": "- Hydrating cleanser, thick moisturizer, avoid hot water",
        "Home Remedies": "- Aloe vera gel\n- Coconut oil massage\n- Honey mask"
    },
    "normal": {
        "Skincare": "- Gentle cleanser, lightweight moisturizer, regular sunscreen",
        "Home Remedies": "- Cucumber slices\n- Rose water toner\n- Balanced diet"
    }
}

# step 1: upload image
uploaded = files.upload()
for fn in uploaded.keys():
    img_path = fn

# step 2: load and preprocess image
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# step 3: prediction
pred = model.predict(x)
class_idx = np.argmax(pred)
confidence = np.max(pred)

# class labels
class_labels = list(train_data.class_indices.keys())
predicted_class = class_labels[class_idx]

# show image + prediction
plt.imshow(plt.imread(img_path))
plt.axis("off")
plt.title(f"Prediction: {predicted_class} (confidence: {confidence:.2f})")
plt.show()

# show remedies
print(f"\nSkin Type: {predicted_class.capitalize()}")
print("Skincare:", remedies[predicted_class]["Skincare"])
print("Home Remedies:", remedies[predicted_class]["Home Remedies"])

model.save("skin_type_model.h5")


