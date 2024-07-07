# AIcreator
 Create Sequential Neural Networks Using a GUI program

Certainly! Here's a README file that explains the program, how to use it, and how to contribute:

# Neural Network GUI

This Neural Network GUI is a Python application that provides a graphical interface for creating, training, and analyzing neural networks using TensorFlow and Keras. It offers a user-friendly way to experiment with different network architectures and parameters without writing code.

## Features

- Load and preprocess CSV data
- Create custom neural network architectures
- Train models with various optimizers and loss functions
- Visualize model performance
- Make predictions using trained models
- Analyze model structure and feature importance
- Save and load trained models
- GPU support (if available)

## Requirements

- Python 3.7+
- PySide6
- TensorFlow 2.x
- Pandas
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/neural-network-gui.git
   cd neural-network-gui
   ```

2. Install the required dependencies:
   ```
    pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python NeuralNetworkCreator.py
   ```

2. Load your CSV data using the "Load CSV Data" button.

3. Configure your neural network:
   - Set input and output dimensions
   - Add layers using the dropdown menus and input fields
   - Set training parameters (epochs, batch size)
   - Choose optimizer and loss function

4. Train the model using the "Train Model" button.

5. Use the "Predict" function to make predictions with your trained model.

6. Analyze your model using the "Analyze Model" button to open the analysis window.

## Contributing

Contributions to improve the Neural Network GUI are welcome!

### Areas for Improvement

- Add support for more layer types and activation functions
- Implement data visualization features
- Enhance the model analysis capabilities
- Improve error handling and user feedback
- Add unit tests and documentation

## License

This project is licensed under the GNU License. See the LICENSE file for details.

## Acknowledgments

This GUI application uses TensorFlow, Keras, and PySide6.
