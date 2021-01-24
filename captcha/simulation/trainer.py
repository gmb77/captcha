import tensorflow.keras as ker


def define_model(label_num):
    return ker.Sequential([
        ker.layers.experimental.preprocessing.Rescaling(1. / 255),
        ker.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        ker.layers.MaxPooling2D(),
        ker.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
        ker.layers.MaxPooling2D(),
        ker.layers.Flatten(),
        ker.layers.Dense(128, activation="relu"),
        ker.layers.Dense(label_num, activation="softmax")
    ])


def do_training(input_dir, model_file, new_size, label_num):
    model = define_model(label_num)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Fetch training set
    training_set = ker.preprocessing.image_dataset_from_directory(
        input_dir, image_size=new_size, label_mode="categorical", color_mode="grayscale", shuffle=True)

    # Train the network and save it
    model.fit(training_set, epochs=5)
    model.save(model_file)
