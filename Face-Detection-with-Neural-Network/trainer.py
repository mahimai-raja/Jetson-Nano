import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

train_data_gen = ImageDataGenerator(rescale=1./255)
valid_data_gen = ImageDataGenerator(rescale=1./255)

train_generatior = train_data_gen.flow_from_directory(
    'data/augumented/train',
    target_size=(48,48),
    batch_size=4,
    color_mode="grayscale",
    class_mode="categorical"
)

valid_generator = valid_data_gen.flow_from_directory(
    'data/augumented/test',
    target_size=(48, 48),
    batch_size=4,
    color_mode="grayscale",
    class_mode='categorical'
)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

cv.ocl.setUseOpenCL(False)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, decay=1e-6),metrics=['accuracy'])

history = model.fit(
    train_generatior,
    steps_per_epoch=1024 // 64,
    epochs=20,
    validation_data=valid_generator,
    validation_steps= 256 // 64
)

model_structure = model.to_json()
with open("model_structure.json","w") as f :
    f.write(model_structure)

model.save_weights('model_weights.h5')

