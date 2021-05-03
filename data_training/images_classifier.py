from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, Flatten, Softmax
from tensorflow.python.keras.regularizers import L1L2

from settings import MODEL_FILENAME


class ImagesClassifier:
    log_name = '[DATA TRAINING][IMAGE CLASSIFIER]'
    # batch_size = 5000
    # batch_size = 1024
    batch_size = 2048

    model_filename = MODEL_FILENAME

    def __init__(self, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set,
                 generator=None):
        self.generator = generator

        self.x_train_set = x_train_set
        self.y_train_set = y_train_set
        self.x_validate_set = x_validate_set
        self.y_validate_set = y_validate_set
        self.x_test_set = x_test_set
        self.y_test_set = y_test_set

        self.__classify()

    def __classify(self):
        print(f'{self.log_name} Starting image classification')

        self.__flatten_datasets()

        model = self.__create_logistic_regression_model()

        self.train_model(model, self.generator, self.x_train_set, self.y_train_set, self.x_validate_set,
                         self.y_validate_set, self.x_test_set, self.y_test_set)

        print(f'{self.log_name} Finished image classification')

    def __flatten_datasets(self):
        print(f'{self.log_name} Flattening datasets')

        # self.x_train_set = self.__flatten_x_dataset(self.x_train_set)
        self.y_train_set = self.__flatten_y_dataset(self.y_train_set)
        # self.x_validate_set = self.__flatten_x_dataset(self.x_validate_set)
        self.y_validate_set = self.__flatten_y_dataset(self.y_validate_set)
        # self.x_test_set = self.__flatten_x_dataset(self.x_test_set)
        self.y_test_set = self.__flatten_y_dataset(self.y_test_set)

    @staticmethod
    def __flatten_x_dataset(dataset):
        samples_number = len(dataset)
        dataset = dataset.reshape((samples_number, -1))

        return dataset

    @staticmethod
    def __flatten_y_dataset(dataset):
        dataset = to_categorical(dataset)
        return dataset

    @staticmethod
    def __create_logistic_regression_model():
        input_shape = (72, 128, 3)
        inputs = tf.keras.Input(shape=input_shape)
        base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
        # base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, weights='imagenet')

        # base_model.trainable = False
        base_model.trainable = True

        # x = base_model(inputs, training=False)
        x = base_model(inputs, training=True)

        x = Flatten()(x)
        # x = ReLU()(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dense(1024)(x)
        # + 2 dense, output-_dim = 1024,

        x = Dense(2,  # output dim is 2, one score per each class
                  # activation='sigmoid',
                  kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                  # input_dim=features_number
                  )(x)

        outputs = Softmax()(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam',  # 'adam'
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy', tf.keras.metrics.BinaryCrossentropy()])

        return model

    def train_model(self, model: Model, generator: ImageDataGenerator, x_train_set, y_train_set, x_val_set, y_val_set,
                    x_test_set, y_test_set):
        print(f'{self.log_name} Started training on training set')
        start_time = timer()

        if generator:
            model, history = self.__fit_model_with_generator(model, generator, x_train_set, y_train_set, x_val_set,
                                                             y_val_set, batch_size=self.batch_size)
        else:
            model, history = self.__fit_model(model, x_train_set, y_train_set, x_val_set, y_val_set,
                                              batch_size=self.batch_size)

        print(f'{self.log_name} Started prediction on testing set')
        self.__evaluate_model(model, x_test_set, y_test_set)
        print(f'{self.log_name} Finished prediction on testing set')

        elapsed_time = timer() - start_time
        print(f'{self.log_name} Finished training on training set with elapsed time: ({elapsed_time})')
        self.__plot_training_statistics(history)
        # self.__save_model(model)

    @staticmethod
    def __fit_model(model: Model, x_train_dataset, y_train_dataset, x_val_dataset, y_val_dataset,
                    batch_size=1000, epochs=20):
        history = model.fit(x_train_dataset, y_train_dataset, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val_dataset, y_val_dataset))
        return model, history

    def __fit_model_with_generator(self, model: Model, generator: ImageDataGenerator, x_train_dataset, y_train_dataset,
                                   x_val_dataset, y_val_dataset, batch_size=1000, epochs=20):
        history = model.fit(generator.flow(x_train_dataset, y_train_dataset, batch_size=batch_size), epochs=epochs,
                            validation_data=(x_val_dataset, y_val_dataset),
                            steps_per_epoch=np.ceil(len(y_train_dataset) / batch_size), shuffle=True)
        return model, history

    @staticmethod
    def __evaluate_model(model: Model, x_dataset, y_dataset):
        model.evaluate(x_dataset, y_dataset)

        # y_dataset = [np.argmax(y) for y in y_dataset]
        # self.__get_classification_report(y_dataset, predictions)
        # self.__plot_confusion_matrix(y_dataset, predictions)

    @staticmethod
    def __convert_prediction_probabilities_to_classes(prediction_probabilities):
        # predictions = [np.argmax(probabilities) for probabilities in prediction_probabilities]
        predictions = np.where(prediction_probabilities > 0.5, 1, 0)
        return np.array(predictions)

    def __get_classification_report(self, y_dataset, predictions):
        print(f'{self.log_name} Classification report:')
        print(f'{metrics.classification_report(y_dataset, predictions)}')

    def __plot_confusion_matrix(self, y_dataset, predictions):
        confusion_matrix = metrics.confusion_matrix(y_dataset, predictions)
        print(f'{self.log_name} Confusion matrix:')
        print(f'{confusion_matrix}')

    @staticmethod
    def __plot_training_statistics(history):
        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('Model accuracy during training')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train set', 'validation set'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss during training')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train set', 'validation set'], loc='upper left')
        plt.show()

    def __save_model(self, model: Model):
        model.save(self.model_filename)

    # def __get_binary_cross_entropy_loss(self, y_dataset, predictions):
    #     cross_entropy = metrics.log_loss(y_dataset, predictions)
    #     print(f'{self.log_name} Binary cross entropy loss is: ({cross_entropy})')
    #     return cross_entropy
    #
    # def __get_accuracy(self, y_dataset, predictions):
    #     accuracy = metrics.accuracy_score(y_dataset, predictions)
    #     print(f'{self.log_name} Accuracy is: ({accuracy})')
    #     return accuracy