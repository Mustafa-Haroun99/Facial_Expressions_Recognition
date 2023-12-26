def input_image_normalization(image, label):
  image = tf.cast(image/255. , tf.float32)
  return image, label


def data_normalization(train_set, validation_set, test_set):
    norm_train_set = train_set.map(input_image_normalization)
    norm_vald_set = validation_set.map(input_image_normalization)
    norm_test_set = test_set.map(input_image_normalization)
    return norm_train_set, norm_vald_set, norm_test_set


def data_augmentation(x_train, y_train):
    data_generator = ImageDataGenerator(rotation_range=10,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=.1,
                                        horizontal_flip=True)
    flow = data_generator.flow(x_train,
                               y_train,
                               batch_size=128)
    return flow