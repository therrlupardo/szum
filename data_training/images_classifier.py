from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, Flatten, Softmax
from tensorflow.python.keras.regularizers import L1L2


class ImagesClassifier:
    log_name = '[DATA TRAINING][IMAGE CLASSIFIER]'

    batch_size = 1024
    epochs = 5

    def __init__(self, model_filename):
        self.model_filename = model_filename
        self.generator = None

        self.x_train_set = None
        self.y_train_set = None
        self.x_validate_set = None
        self.y_validate_set = None
        self.x_test_set = None
        self.y_test_set = None

    def create_model(self, x_train_set, y_train_set, x_validate_set, y_validate_set, x_test_set, y_test_set,
                     generator=None):
        self.generator = generator

        self.x_train_set = x_train_set
        self.y_train_set = y_train_set
        self.x_validate_set = x_validate_set
        self.y_validate_set = y_validate_set
        self.x_test_set = x_test_set
        self.y_test_set = y_test_set

        model = self.__create_classification_model()
        self.__save_model(model)

    def use_model(self, x_dataset, y_dataset):
        model = self.__load_model()
        self.__use_classification_model(model, x_dataset, y_dataset)
        pass

    def __create_classification_model(self):
        print(f'{self.log_name} Starting image classification')

        self.__flatten_datasets()

        input_shape = self.__get_input_dataset_shape(self.x_train_set)
        model = self.__create_model(input_shape)

        model = self.__train_model(model, self.generator, self.x_train_set, self.y_train_set, self.x_validate_set,
                                   self.y_validate_set, self.x_test_set, self.y_test_set)

        print(f'{self.log_name} Finished image classification')
        return model

    def __use_classification_model(self, model: Model, x_dataset, y_dataset):
        print(f'{self.log_name} Starting test image classification')

        y_dataset = self.__flatten_y_dataset(y_dataset)

        print(f'{self.log_name} Started prediction on loaded model')
        self.__evaluate_model(model, x_dataset, y_dataset)
        self.__predict_model(model, x_dataset, y_dataset)
        print(f'{self.log_name} Finished prediction on loaded model')

        print(f'{self.log_name} Finished test image classification')

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
    def __get_input_dataset_shape(x_dataset: np.array):
        _, x, y, z = x_dataset.shape
        input_shape = (x, y, z)

        return input_shape

    @staticmethod
    def __create_model(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')

        base_model.trainable = True
        x = base_model(inputs, training=True)

        x = Flatten()(x)
        x = Dense(2, kernel_regularizer=L1L2(l1=0.0, l2=0.1))(x)
        outputs = Softmax()(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.BinaryCrossentropy(),
                               tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model

    def __train_model(self, model: Model, generator: ImageDataGenerator, x_train_set, y_train_set, x_val_set, y_val_set,
                      x_test_set, y_test_set):
        print(f'{self.log_name} Started training on training set')
        start_time = timer()

        if generator:
            model, history = self.__fit_model_with_generator(model, generator, x_train_set, y_train_set, x_val_set,
                                                             y_val_set, batch_size=self.batch_size, epochs=self.epochs)
        else:
            model, history = self.__fit_model(model, x_train_set, y_train_set, x_val_set, y_val_set,
                                              batch_size=self.batch_size, epochs=self.epochs)

        print(f'{self.log_name} Started prediction on testing set')
        self.__evaluate_model(model, x_test_set, y_test_set)
        self.__predict_model(model, x_test_set, y_test_set)
        print(f'{self.log_name} Finished prediction on testing set')

        elapsed_time = timer() - start_time
        print(
            f'{self.log_name} Finished training on training set with elapsed time: ({elapsed_time:0.2f} seconds) or ({elapsed_time / 60:0.2f} minutes)')
        self.__plot_training_statistics(history)

        return model

    @staticmethod
    def __fit_model(model: Model, x_train_dataset, y_train_dataset, x_val_dataset, y_val_dataset,
                    batch_size=1024, epochs=20):
        history = model.fit(x_train_dataset, y_train_dataset, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val_dataset, y_val_dataset))
        return model, history

    @staticmethod
    def __fit_model_with_generator(model: Model, generator: ImageDataGenerator, x_train_dataset, y_train_dataset,
                                   x_val_dataset, y_val_dataset, batch_size=1024, epochs=20):
        history = model.fit(generator.flow(x_train_dataset, y_train_dataset, batch_size=batch_size), epochs=epochs,
                            validation_data=(x_val_dataset, y_val_dataset),
                            steps_per_epoch=np.ceil(len(y_train_dataset) / batch_size), shuffle=True)
        return model, history

    @staticmethod
    def __evaluate_model(model: Model, x_dataset, y_dataset):
        model.evaluate(x_dataset, y_dataset)

    def __predict_model(self, model: Model, x_dataset, y_dataset):
        prediction_probabilities = model.predict(x_dataset)

        predictions = self.__convert_prediction_probabilities_to_classes(prediction_probabilities)

        y_dataset = [np.argmax(y) for y in y_dataset]
        predictions = [np.argmax(prediction) for prediction in predictions]
        self.__get_classification_report(y_dataset, predictions)
        self.__get_confusion_matrix(y_dataset, predictions)
        self.__plot_receiver_operating_characteristic(y_dataset, predictions)
        self.__get_model_metrics_from_predictions(y_dataset, predictions)

    @staticmethod
    def __convert_prediction_probabilities_to_classes(prediction_probabilities):
        predictions = np.where(prediction_probabilities > 0.5, 1, 0)
        return np.array(predictions)

    def __get_classification_report(self, y_dataset, predictions):
        print(f'{self.log_name} Classification report:')
        print(f'{metrics.classification_report(y_dataset, predictions)}')

    def __get_confusion_matrix(self, y_dataset, predictions):
        confusion_matrix = metrics.confusion_matrix(y_dataset, predictions)
        print(f'{self.log_name} Confusion matrix:')
        print(f'{confusion_matrix}')

    def __get_model_metrics_from_predictions(self, y_dataset, predictions):
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()

        precision_metric.update_state(y_dataset, predictions)
        recall_metric.update_state(y_dataset, predictions)
        _, _, area_under_curve_metric = self.__calculate_receiver_operating_characteristic(y_dataset, predictions)

        precision_metric = precision_metric.result().numpy()
        recall_metric = recall_metric.result().numpy()

        f2_score_metric = self.__calculate_f_score_metric(precision_metric, recall_metric, beta=2)

        print(f'{self.log_name} Precision: {precision_metric}')
        print(f'{self.log_name} Recall: {recall_metric}')
        print(f'{self.log_name} F2 score: {f2_score_metric}')
        print(f'{self.log_name} Area under curve: {area_under_curve_metric:0.3f}')

    @staticmethod
    def __calculate_f_score_metric(precision, recall, beta=1):
        f_score = (1 + beta * beta) * (precision * recall) / (beta * beta * precision + recall)
        return f_score

    @staticmethod
    def __calculate_receiver_operating_characteristic(y_dataset, predictions):
        false_positive_rates, true_positive_rates, _ = roc_curve(y_dataset, predictions)
        area_under_curve = auc(false_positive_rates, true_positive_rates)

        return false_positive_rates, true_positive_rates, area_under_curve

    def __plot_receiver_operating_characteristic(self, y_dataset, predictions):
        false_positive_rates, true_positive_rates, area_under_curve = self.__calculate_receiver_operating_characteristic(
            y_dataset, predictions)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(false_positive_rates, true_positive_rates,
                 label=f'model ROC (area under curve = {area_under_curve:0.3f})')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('Model ROC curve during testing')
        plt.legend(loc='best')
        plt.show()

    def __plot_training_statistics(self, history):
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

        precision = history.history['precision']
        val_precision = history.history['val_precision']
        recall = history.history['recall']
        val_recall = history.history['val_recall']

        plt.plot(precision)
        plt.plot(val_precision)
        plt.title('Model precision during training')
        plt.ylabel('precision')
        plt.xlabel('epoch')
        plt.legend(['train set', 'validation set'], loc='upper left')
        plt.show()

        plt.plot(recall)
        plt.plot(val_recall)
        plt.title('Model recall during training')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['train set', 'validation set'], loc='upper left')
        plt.show()

        f2_score = self.__calculate_list_of_f_score_metrics(precision, recall, beta=2)
        val_f2_score = self.__calculate_list_of_f_score_metrics(val_precision, val_recall, beta=2)

        plt.plot(f2_score)
        plt.plot(val_f2_score)
        plt.title('Model f2-score during training')
        plt.ylabel('f2-score')
        plt.xlabel('epoch')
        plt.legend(['train set', 'validation set'], loc='upper left')
        plt.show()

    def __calculate_list_of_f_score_metrics(self, precision_list, recall_list, beta=1):
        f_score_list = []
        for i in range(len(precision_list)):
            result = self.__calculate_f_score_metric(precision_list[i], recall_list[i], beta)
            f_score_list.append(result)

        return f_score_list

    def __save_model(self, model: Model):
        model.save(self.model_filename)
        print(f'{self.log_name} Saved model {self.model_filename}')

    def __load_model(self):
        model = tf.keras.models.load_model(self.model_filename)
        print(f'{self.log_name} Loaded model {self.model_filename}')
        return model
