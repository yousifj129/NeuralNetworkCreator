import pandas as pd
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QLabel, 
                             QFileDialog, QTextEdit, QMessageBox,QTabWidget)
from PySide6.QtCharts import QChart, QChartView, QLineSeries
from PySide6.QtGui import QPainter


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
