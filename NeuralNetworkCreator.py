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


class AnalyzeWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        hideButton = QPushButton("Hide")
        hideButton.clicked.connect(self.hide)
        layout.addWidget(hideButton)
        # Tabs for different analysis options
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # Model Overview Tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        overview_layout.addWidget(QLabel("Model Overview:"))
        overview_layout.addWidget(self.model_info)
        tab_widget.addTab(overview_tab, "Model Overview")

        # Visualization Tab
        visualization_tab = QWidget()
        visualization_layout = QVBoxLayout(visualization_tab)

        # Add input fields for linspace parameters
        linspace_layout = QHBoxLayout()
        self.start_input = QLineEdit("-10")
        self.end_input = QLineEdit("10")
        self.num_points_input = QLineEdit("200")
        update_btn = QPushButton("Update Visualization")
        update_btn.clicked.connect(self.update_visualization)

        linspace_layout.addWidget(QLabel("Start:"))
        linspace_layout.addWidget(self.start_input)
        linspace_layout.addWidget(QLabel("End:"))
        linspace_layout.addWidget(self.end_input)
        linspace_layout.addWidget(QLabel("Points:"))
        linspace_layout.addWidget(self.num_points_input)
        linspace_layout.addWidget(update_btn)

        visualization_layout.addLayout(linspace_layout)

        self.chart_view = QChartView()
        visualization_layout.addWidget(self.chart_view)
        tab_widget.addTab(visualization_tab, "Visualization")

        # Test Data Tab
        test_data_tab = QWidget()
        test_data_layout = QVBoxLayout(test_data_tab)
        load_test_btn = QPushButton("Load Test Data")
        load_test_btn.clicked.connect(self.load_test_data)
        test_data_layout.addWidget(load_test_btn)
        self.test_results = QTextEdit()
        self.test_results.setReadOnly(True)
        test_data_layout.addWidget(self.test_results)
        tab_widget.addTab(test_data_tab, "Test Data")

        # Feature Importance Tab
        feature_importance_tab = QWidget()
        feature_importance_layout = QVBoxLayout(feature_importance_tab)
        analyze_features_btn = QPushButton("Analyze Feature Importance")
        analyze_features_btn.clicked.connect(self.analyze_feature_importance)
        feature_importance_layout.addWidget(analyze_features_btn)
        self.feature_importance_results = QTextEdit()
        self.feature_importance_results.setReadOnly(True)
        feature_importance_layout.addWidget(self.feature_importance_results)
        tab_widget.addTab(feature_importance_tab, "Feature Importance")

        # Update information
        self.update_model_info()
        self.visualize_model()

        self.setWindowTitle('Model Analysis')
        self.setGeometry(300, 300, 600, 400)
        self.show()
        
    def update_model_info(self):
        if self.main_window.model:
            info = f"Model Summary:\n{self.get_model_summary()}\n\n"
            info += f"Input Shape: {self.main_window.model.input_shape}\n"
            info += f"Output Shape: {self.main_window.model.output_shape}\n"
            info += f"Total Parameters: {self.main_window.model.count_params()}\n"
            info += f"Optimizer: {self.main_window.model.optimizer.__class__.__name__}\n"
            info += f"Loss Function: {self.main_window.model.loss}\n"
            info += f"Model Config:\n\n\n\n\n{self.main_window.model.get_config()}\n\n"
            self.model_info.setText(info)
        else:
            self.model_info.setText("No model loaded.")

    def get_model_summary(self):
        stringlist = []
        self.main_window.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)

    def visualize_model(self):
        if self.main_window.model and self.main_window.model.input_shape[1] == 1 and self.main_window.model.output_shape[1] == 1:
            start = float(self.start_input.text())
            end = float(self.end_input.text())
            num_points = int(self.num_points_input.text())

            chart = QChart()
            series = QLineSeries()

            x = np.linspace(start, end, num_points)
            y = self.main_window.model.predict(x.reshape(-1, 1)).flatten()

            for i in range(len(x)):
                series.append(x[i], y[i])

            chart.addSeries(series)
            chart.createDefaultAxes()
            chart.setTitle("Model Visualization")

            self.chart_view.setChart(chart)
            self.chart_view.setRenderHint(QPainter.Antialiasing)
        else:
            self.chart_view.setChart(QChart())  # Clear the chart

    def update_visualization(self):
        try:
            self.visualize_model()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for start, end, and number of points.")

    def load_test_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Test Data", "", "CSV Files (*.csv)")
        if file_name:
            try:
                test_data = pd.read_csv(file_name)
                x_test = test_data.iloc[:, :-1].values
                y_test = test_data.iloc[:, -1].values
                
                loss, mae = self.main_window.model.evaluate(x_test, y_test)
                self.test_results.setText(f"Test Loss: {loss:.4f}\nTest MAE: {mae:.4f}")
            except Exception as e:
                self.test_results.setText(f"Error loading test data: {str(e)}")

    def analyze_feature_importance(self):
        if self.main_window.model and self.main_window.data is not None:
            x = self.main_window.data.iloc[:, :-1].values
            y = self.main_window.data.iloc[:, -1].values
            
            base_mae = self.main_window.model.evaluate(x, y)[1]
            
            importance = []
            for i in range(x.shape[1]):
                x_permuted = x.copy()
                x_permuted[:, i] = np.random.permutation(x_permuted[:, i])
                mae = self.main_window.model.evaluate(x_permuted, y)[1]
                importance.append((mae - base_mae) / base_mae)
            
            feature_names = self.main_window.data.columns[:-1]
            importance_str = "Feature Importance:\n"
            for name, imp in zip(feature_names, importance):
                importance_str += f"{name}: {imp:.4f}\n"
            
            self.feature_importance_results.setText(importance_str)
        else:
            self.feature_importance_results.setText("Model or data not available for analysis.")

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

        self.use_gpu_checkbox = QCheckBox("Use GPU (if available)")
        self.use_gpu_checkbox.setChecked(self.check_gpu_availability())
        left_layout.addWidget(self.use_gpu_checkbox)

        self.clear_model_checkbox = QCheckBox("Clear model before training")
        self.clear_model_checkbox.setChecked(True)
        left_layout.addWidget(self.clear_model_checkbox)

        # Data Loading
        load_data_btn = QPushButton("Load CSV Data")
        load_data_btn.clicked.connect(self.load_data)
        left_layout.addWidget(load_data_btn)

        # Model Configuration
        self.input_dim_edit = QLineEdit()
        self.output_dim_edit = QLineEdit()
        left_layout.addWidget(QLabel("Input Dimension:"))
        left_layout.addWidget(self.input_dim_edit)
        left_layout.addWidget(QLabel("Output Dimension:"))
        left_layout.addWidget(self.output_dim_edit)

        # Layer Configuration
        self.layer_type = QComboBox()
        self.layer_type.addItems(["Dense","Dropout"])
        self.layer_param = QLineEdit()
        left_layout.addWidget(QLabel("Layer Type:"))
        left_layout.addWidget(self.layer_type)
        left_layout.addWidget(QLabel("Layer Parameters:"))
        left_layout.addWidget(self.layer_param)
        self.activation = QComboBox()
        self.activation.addItems(["relu", "sigmoid", "softmax", "tanh", "linear", 
                                  "softplus","softsign","selu","elu","exponential","leaky_relu",
                                  "relu6","silu","hard_silu","gelu","hard_sigmoid","mish","log_softmax"])
        left_layout.addWidget(QLabel("Activation Function:"))
        left_layout.addWidget(self.activation)
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer)
        left_layout.addWidget(add_layer_btn)

        # Layer List
        self.layer_list = QListWidget()
        self.layer_list.itemClicked.connect(self.remove_layer)
        left_layout.addWidget(QLabel("Layers (click to remove):"))
        left_layout.addWidget(self.layer_list)

        # Training Configuration
        self.epochs = QLineEdit()
        self.batch_size = QLineEdit()
        left_layout.addWidget(QLabel("Epochs:"))
        left_layout.addWidget(self.epochs)
        left_layout.addWidget(QLabel("Batch Size:"))
        left_layout.addWidget(self.batch_size)

        # Optimizer and Loss Function
        self.optimizer = QComboBox()
        self.optimizer.addItems(["adam", "sgd", "rmsprop", "adagrad","adamw","adadelta","adamax","adafactor","nadam","ftrl","lion"])
        self.loss_function = QComboBox()
        self.loss_function.addItems(["mean_squared_error", "mean_absolute_error", "categorical_crossentropy","binary_crossentropy"
                                        ,"mean_absolute_percentage_error","mean_squared_logarithmic_error"
                                      , "cosine_similarity","huber","log_cosh","hinge"
                                      ,"categorical_hinge","squared_hinge","kl_divergence","poisson","sparse_categorical_crossentropy"])

        left_layout.addWidget(QLabel("Optimizer:"))
        left_layout.addWidget(self.optimizer)
        left_layout.addWidget(QLabel("Loss Function:"))
        left_layout.addWidget(self.loss_function)
        

        # Buttons
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)
        left_layout.addWidget(train_btn)

        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        left_layout.addWidget(load_btn)

        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model)
        left_layout.addWidget(save_btn)

        clear_btn = QPushButton("Clear Model")
        clear_btn.clicked.connect(self.clear_model)
        left_layout.addWidget(clear_btn)

        analyze_btn = QPushButton("Analyze Model")
        analyze_btn.clicked.connect(self.openAnalyzeWindow)
        left_layout.addWidget(analyze_btn)

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

        if self.use_gpu_checkbox.isChecked():
            with tf.device('/GPU:0'):
                self.create_model()
                self.compile_and_train_model(epochs, batch_size)
        else:
            with tf.device('/CPU:0'):
                self.create_model()
                self.compile_and_train_model(epochs, batch_size)

    def compile_and_train_model(self, epochs, batch_size):
        optimizer = self.optimizer.currentText()
        loss = self.loss_function.currentText()

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
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Keras Model (*.h5)")
        if file_name:
            try:
                self.model = keras.models.load_model(file_name)
                self.input_dim = self.model.input_shape[1]
                self.output_dim = self.model.output_shape[1]
                self.input_dim_edit.setText(str(self.input_dim))
                self.output_dim_edit.setText(str(self.output_dim))
                self.layer_list.clear()
                for layer in self.model.layers[1:-1]:
                    self.layer_list.addItem(f"{layer.__class__.__name__}: {layer.get_config()}")
                self.loss_display.setText("Model loaded successfully")
            except Exception as e:
                self.loss_display.setText(f"Error loading model: {str(e)}")

    def save_model(self):
        if self.model:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Keras Model (*.h5)")
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