import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QLabel, QListWidget, QComboBox, 
                             QFileDialog, QTextEdit, QMessageBox,QCheckBox,QTabWidget)
from PySide6.QtCharts import QChart, QChartView, QLineSeries
from PySide6.QtGui import QPainter
import tensorflow as tf
from tensorflow import keras
from AnalyzeWindow import AnalyzeWindow


class NeuralNetworkGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.layers = []
        self.data = None
        self.input_dim = 0
        self.output_dim = 0
        self.init_ui()

    def init_ui(self):
        self.analyze_window = AnalyzeWindow(self)
        self.analyze_window.hide()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        checkbox_layout = QHBoxLayout()

        self.use_gpu_checkbox = QCheckBox("Use GPU (if available)")
        self.use_gpu_checkbox.setChecked(self.check_gpu_availability())
        checkbox_layout.addWidget(self.use_gpu_checkbox)

        self.clear_model_checkbox = QCheckBox("New model on train (clear old model)")
        self.clear_model_checkbox.setChecked(False)
        checkbox_layout.addWidget(self.clear_model_checkbox)

        left_layout.addLayout(checkbox_layout)
        # Data Loading
        load_data_btn = QPushButton("Load CSV Data")
        load_data_btn.clicked.connect(self.load_data)
        left_layout.addWidget(load_data_btn)

        # Model Configuration
        input_output_layout = QHBoxLayout()
        self.input_dim_edit = QLineEdit()
        self.input_dim_edit.setText("1")
        self.output_dim_edit = QLineEdit()
        self.output_dim_edit.setText("1")
        input_output_layout.addWidget(QLabel("Input Dim:"))
        input_output_layout.addWidget(self.input_dim_edit)
        input_output_layout.addWidget(QLabel("Output Dim:"))
        input_output_layout.addWidget(self.output_dim_edit)
        left_layout.addLayout(input_output_layout)

        # Layer Configuration
        layer_layout = QHBoxLayout()
        self.layer_type = QComboBox()
        self.layer_type.addItems(["Dense","Dropout"])
        self.layer_param = QLineEdit()
        self.layer_param.setText("25")
        layer_layout.addWidget(QLabel("Layer Type:"))
        layer_layout.addWidget(self.layer_type)
        layer_layout.addWidget(QLabel("Layer Parameters:"))
        layer_layout.addWidget(self.layer_param)
        self.activation = QComboBox()
        self.activation.addItems(["relu", "sigmoid", "softmax", "tanh", "linear", 
                                  "softplus","softsign","selu","elu","exponential","leaky_relu",
                                  "relu6","silu","hard_silu","gelu","hard_sigmoid","mish","log_softmax"])
        layer_layout.addWidget(QLabel("Activation Function:"))
        layer_layout.addWidget(self.activation)
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer)
        left_layout.addLayout(layer_layout)
        left_layout.addWidget(add_layer_btn)

        # Layer List
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.remove_layer)
        left_layout.addWidget(QLabel("Layers (click to remove):"))
        left_layout.addWidget(self.layer_list)

        # Training Configuration
        train_layout = QHBoxLayout()
        self.epochs = QLineEdit()
        self.epochs.setText("100")
        self.batch_size = QLineEdit()
        self.batch_size.setText("8")
        train_layout.addWidget(QLabel("Epochs:"))
        train_layout.addWidget(self.epochs)
        train_layout.addWidget(QLabel("Batch Size:"))
        train_layout.addWidget(self.batch_size)
        left_layout.addLayout(train_layout)
        # Optimizer and Loss Function
        optimizer_layout = QHBoxLayout()
        self.optimizer = QComboBox()
        self.optimizer.addItems(["adam", "sgd", "rmsprop", "adagrad","adamw","adadelta","adamax","adafactor","nadam","ftrl","lion"])
        self.loss_function = QComboBox()
        self.loss_function.addItems(["mean_squared_error", "mean_absolute_error", "categorical_crossentropy","binary_crossentropy"
                                        ,"mean_absolute_percentage_error","mean_squared_logarithmic_error"
                                      , "cosine_similarity","huber","log_cosh","hinge"
                                      ,"categorical_hinge","squared_hinge","kl_divergence","poisson","sparse_categorical_crossentropy"])

        optimizer_layout.addWidget(QLabel("Optimizer:"))
        optimizer_layout.addWidget(self.optimizer)
        optimizer_layout.addWidget(QLabel("Loss Function:"))
        optimizer_layout.addWidget(self.loss_function)

        left_layout.addLayout(optimizer_layout)

        # Buttons
        buttons1_layout = QHBoxLayout()
        buttons2_layout = QHBoxLayout()
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)
        buttons1_layout.addWidget(train_btn)

        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        buttons1_layout.addWidget(load_btn)

        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model)
        buttons1_layout.addWidget(save_btn)

        clear_btn = QPushButton("Clear Model")
        clear_btn.clicked.connect(self.clear_model)
        buttons2_layout.addWidget(clear_btn)

        analyze_btn = QPushButton("Analyze Model")
        analyze_btn.clicked.connect(self.openAnalyzeWindow)
        buttons2_layout.addWidget(analyze_btn)

        left_layout.addLayout(buttons1_layout)
        left_layout.addLayout(buttons2_layout)

        # Prediction
        self.prediction_input = QLineEdit()
        self.prediction_result = QLabel("Prediction: N/A")
        right_layout.addWidget(QLabel("Input for Prediction:"))
        right_layout.addWidget(self.prediction_input)
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.predict)
        right_layout.addWidget(predict_btn)
        right_layout.addWidget(self.prediction_result)

        # Loss Display
        self.loss_display = QTextEdit()
        self.loss_display.setReadOnly(True)
        right_layout.addWidget(QLabel("Training Log:"))
        right_layout.addWidget(self.loss_display)
        self.setGeometry(300, 300, 600, 500)
        self.setWindowTitle('Neural Network GUI')
        self.show()

    def check_gpu_availability(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                return True
            except RuntimeError as e:
                print(e)
        return False
    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load CSV Data", "", "CSV Files (*.csv)")
        if file_name:
            try:
                self.data = pd.read_csv(file_name)
                self.input_dim = self.data.shape[1] - 1  # Assuming last column is the target
                self.output_dim = 1  # Assuming single target variable
                self.input_dim_edit.setText(str(self.input_dim))
                self.output_dim_edit.setText(str(self.output_dim))
                self.loss_display.setText(f"Data loaded successfully. Shape: {self.data.shape}")
            except Exception as e:
                self.loss_display.setText(f"Error loading data: {str(e)}")

    def add_layer(self):
        layer_type = self.layer_type.currentText()
        params = self.layer_param.text()
        layer_str = f"{layer_type}: {params}"
        self.layer_list.addItem(layer_str)
    def remove_layer(self, item):
            reply = QMessageBox.question(self, 'Remove Layer', 
                                        "Are you sure you want to remove this layer?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                row = self.layer_list.row(item)
                self.layer_list.takeItem(row)
    def create_model(self):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Input(shape=(self.input_dim,)))
        activ = self.activation.currentText()
        for i in range(self.layer_list.count()):
            layer_str = self.layer_list.item(i).text()
            layer_type, params = layer_str.split(': ')
            
            if layer_type == "Dense":
                units = int(params)
                self.model.add(keras.layers.Dense(units, activation=activ))
            elif layer_type == "Conv1D":
                filters, kernel_size = map(int, params.split(','))
                self.model.add(keras.layers.Conv1D(filters, kernel_size, activation=activ))
            elif layer_type == "LSTM":
                units = int(params)
                self.model.add(keras.layers.LSTM(units, activation=activ))
            elif layer_type == "Dropout":
                rate = float(params)
                self.model.add(keras.layers.Dropout(rate))

        self.model.add(keras.layers.Dense(self.output_dim))

    def train_model(self):
        if self.data is None:
            self.loss_display.setText("Please load data first.")
            return

        self.input_dim = int(self.input_dim_edit.text())
        self.output_dim = int(self.output_dim_edit.text())
        epochs = int(self.epochs.text())
        batch_size = int(self.batch_size.text())

        if self.clear_model_checkbox.isChecked():
            self.create_model()
        
        if self.use_gpu_checkbox.isChecked():
            with tf.device('/GPU:0'):
                self.compile_and_train_model(epochs, batch_size)
        else:
            with tf.device('/CPU:0'):
                self.compile_and_train_model(epochs, batch_size)

    def compile_and_train_model(self, epochs, batch_size):
        optimizer = self.optimizer.currentText()
        loss = self.loss_function.currentText()
        if self.clear_model_checkbox.isChecked():
            self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        if self.model is None:
            self.create_model()
            self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])


        x_train = self.data.iloc[:, :-1].values
        y_train = self.data.iloc[:, -1].values
        class LossHistory(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                self.parent.loss_display.append(f"Epoch {epoch+1}/{epochs} - loss: {logs.get('loss'):.4f} - mae: {logs.get('mae'):.4f}")
                if not self.parent.analyze_window.isHidden():
                    self.parent.analyze_window.update_visualization()
                QApplication.processEvents()  # Update GUI

        history = LossHistory()
        history.parent = self

        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[history])

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Keras Model (*.keras)")
        if file_name:
            try:
                self.model = keras.models.load_model(file_name)
                self.input_dim = self.model.input_shape[1]
                self.output_dim = self.model.output_shape[1]
                self.input_dim_edit.setText(str(self.input_dim))
                self.output_dim_edit.setText(str(self.output_dim))
                self.layer_list.clear()
                for layer in self.model.layers[1:-1]:
                    self.layer_list.addItem(f"{layer.__class__.__name__}: {layer.get_config()['units']}")
                self.loss_display.setText("Model loaded successfully")
            except Exception as e:
                self.loss_display.setText(f"Error loading model: {str(e)}")

    def save_model(self):
        if self.model:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Keras Model (*.keras)")
            if file_name:
                self.model.save(file_name)
                self.loss_display.setText("Model saved successfully")
        else:
            self.loss_display.setText("No model to save")

    def predict(self):
        if self.model is None:
            self.prediction_result.setText("Train or load a model first!")
            return

        try:
            input_data = np.array([float(x.strip()) for x in self.prediction_input.text().split(',')])
            if input_data.shape[0] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} input features, but got {input_data.shape[0]}")
            input_data = input_data.reshape((1, -1))
            prediction = self.model.predict(input_data)
            self.prediction_result.setText(f"Prediction: {prediction[0]}")
        except ValueError as e:
            self.prediction_result.setText(f"Invalid input: {str(e)}")

    def clear_model(self):
        self.model = None
        self.layers = []
        self.data = None
        self.input_dim = 0
        self.output_dim = 0
        self.input_dim_edit.clear()
        self.output_dim_edit.clear()
        self.layer_param.clear()
        self.epochs.clear()
        self.batch_size.clear()
        self.layer_list.clear()
        self.prediction_input.clear()
        self.loss_display.clear()
        self.prediction_result.setText("Prediction: N/A")

    def openAnalyzeWindow(self):
        if self.analyze_window is None:
            self.analyze_window = AnalyzeWindow(self)
        self.analyze_window.show()
        self.analyze_window.update_model_info()

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        ex = NeuralNetworkGUI()
        ex.show()  # Explicitly show the main window
        sys.exit(app.exec())
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()