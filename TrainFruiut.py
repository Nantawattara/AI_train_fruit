from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the VGG16 model
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(128, activation='relu')(x)
output = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.inputs, outputs=output)

# Configure the optimizer
opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Use VGG16's preprocessing
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training data generator
train_it = train_datagen.flow_from_directory(
    'D:\\scr\\AI_train_fruit\\Download_image_Auto\\train',
    class_mode='categorical',
    batch_size=16,
    target_size=(224, 224),
    subset='training'
)

# Validation data generator
val_it = train_datagen.flow_from_directory(
    'D:\\scr\\AI_train_fruit\\Download_image_Auto\\test',
    class_mode='categorical',
    batch_size=16,
    target_size=(224, 224),
    subset='validation'
)

# Set up callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', 
                             save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    train_it,
    steps_per_epoch=len(train_it),
    validation_data=val_it,
    validation_steps=len(val_it),
    epochs=90,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.savefig('training_history.png')

# Save the model
model.save('D:/scr/AI_train_fruit/thaifruitmodel.h5')

# Fine-tuning (optional second phase)
# Unfreeze some top layers for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Continue training with unfrozen layers
history_fine = model.fit(
    train_it,
    steps_per_epoch=len(train_it),
    validation_data=val_it,
    validation_steps=len(val_it),
    epochs=30,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)