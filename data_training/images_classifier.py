from timeit import default_timer as timer

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import L1L2
import tensorflow as tf


class ImagesClassifier:
    log_name = '[DATA TRAINING][IMAGE CLASSIFIER]'
    batch_size = 5000

    def __init__(self, x_validate_set, y_validate_set, x_test_set, y_test_set, x_train_set=None, y_train_set=None,
                 generator=None):
        if generator is None:
            self.x_train_set = x_train_set
            self.y_train_set = y_train_set
        else:
            self.generator = generator

        self.x_validate_set = x_validate_set
        self.y_validate_set = y_validate_set
        self.x_test_set = x_test_set
        self.y_test_set = y_test_set

        self.__classify()

    def __classify(self):
        print(f'{self.log_name} Starting image classification')

        self.__flatten_datasets()

        features_number = len(self.x_test_set[0])
        classifier = self.__create_logistic_regression_model(features_number)

        self.train_model(classifier, self.x_train_set, self.y_train_set, self.x_validate_set, self.y_validate_set,
                         self.x_test_set, self.y_test_set)

        print(f'{self.log_name} Finished image classification')

    def __flatten_datasets(self):
        print(f'{self.log_name} Flattening datasets')

        self.x_train_set = self.__flatten_x_dataset(self.x_train_set)
        self.y_train_set = self.__flatten_y_dataset(self.y_train_set)
        self.x_validate_set = self.__flatten_x_dataset(self.x_validate_set)
        self.y_validate_set = self.__flatten_y_dataset(self.y_validate_set)
        self.x_test_set = self.__flatten_x_dataset(self.x_test_set)
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
    def __create_logistic_regression_model(features_number):
        model = Sequential()
        model.add(Dense(2,  # output dim is 2, one score per each class
                        activation='sigmoid',
                        kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                        input_dim=features_number))

        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy', tf.keras.metrics.BinaryCrossentropy()])

        return model

    def train_model(self, model, x_train_set, y_train_set, x_val_set, y_val_set, x_test_set, y_test_set):
        print(f'{self.log_name} Started training on training set')
        start_time = timer()

        model, history = self.__fit_model(model, x_train_set, y_train_set, x_val_set, y_val_set,
                                          batch_size=self.batch_size)

        print(f'{self.log_name} Started prediction on testing set')
        self.__evaluate_model(model, x_test_set, y_test_set)
        print(f'{self.log_name} Finished prediction on testing set')

        elapsed_time = timer() - start_time
        print(f'{self.log_name} Finished training on training set with elapsed time: ({elapsed_time})')
        self.__plot_training_statistics(history)

    @staticmethod
    def __fit_model(model: tf.keras.Model, x_train_dataset, y_train_dataset, x_val_dataset, y_val_dataset,
                    batch_size=None, epochs=100):
        history = model.fit(x_train_dataset, y_train_dataset, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val_dataset, y_val_dataset))
        return model, history

    @staticmethod
    def __fit_model_with_generator(model: tf.keras.Model, generator, x_val_dataset, y_val_dataset, batch_size=1000,
                                   epochs=20):
        model.fit(generator, epochs=epochs, batch_size=batch_size, validation_data=(x_val_dataset, y_val_dataset))
        return model

    def __evaluate_model(self, model: tf.keras.Model, x_dataset, y_dataset):
        np.array(model.evaluate(x_dataset, y_dataset))
        # predictions = self.__convert_prediction_probabilities_to_classes(prediction_probabilities)

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

    # def __get_binary_cross_entropy_loss(self, y_dataset, predictions):
    #     cross_entropy = metrics.log_loss(y_dataset, predictions)
    #     print(f'{self.log_name} Binary cross entropy loss is: ({cross_entropy})')
    #     return cross_entropy
    #
    # def __get_accuracy(self, y_dataset, predictions):
    #     accuracy = metrics.accuracy_score(y_dataset, predictions)
    #     print(f'{self.log_name} Accuracy is: ({accuracy})')
    #     return accuracy
