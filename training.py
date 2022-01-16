import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import argparse
from config_reader import read_config


def Dataloader(data_dir,process_configs):
    batch_size = process_configs['batch_size']
    img_height = process_configs['img_height']
    img_width = process_configs['img_width']
    # data_dir = "/content/drive/MyDrive/Face_mask_classification/PNG_Data/"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=process_configs['validation_split'],
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary',
        labels='inferred',
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=process_configs['validation_split'],
        subset="validation",
        seed=123,
        label_mode='binary',
        labels='inferred',
        image_size=(img_height, img_width),
        shuffle=True,
        batch_size=batch_size)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.2),

        ]
)
    class_names = train_ds.class_names
    print("Classes Names ==>",class_names)
    return train_ds,val_ds

def Visualizer(dataloader):
    plt.figure(figsize=(10, 10))
    for images, labels in dataloader.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            # plt.title(class_names[labels[i]])
            plt.axis("off")

def trainer(trian_loader,valid_loader,process_configs):
    AUTOTUNE = tf.data.AUTOTUNE
    img_height = process_configs['img_height']
    img_width = process_configs['img_width']
    train_ds = trian_loader.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = valid_loader.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    model = Sequential()
    model.add(layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(128, 3, padding="same", activation="relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(512, 3, padding="same", activation="relu",
                            kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.001)))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(process_configs['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    epochs = process_configs['epochs']
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    model.save("1_model.h5")
    print("Training Completed! - Task-2 Done! ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument("--data_dir", type=str, help='Path for Model')
    parser.add_argument("--yaml_file_path", type=str, help='Path for yml file')
    parser.add_argument("--viz", type=bool, help='Visualisation True Or False',default=False)

    args = parser.parse_args()
    process_configs = read_config(file_path=args.yaml_file_path)
    train_ds,val_ds = Dataloader(args.data_dir,process_configs)

    if args.viz:
        Visualizer(train_ds)

    trainer(train_ds,val_ds,process_configs)


