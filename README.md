# Neural Network Creator
 Create Sequential Neural Networks Using a GUI program

## Features

- Load and preprocess CSV data
- Create custom neural network architectures
- Train models with various optimizers and loss functions
- Visualize model performance
- Make predictions using trained models
- Analyze model structure and feature importance
- Save and load trained models
- GPU support (if available)

## note
this program only works with 1D datasets

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yousifj129/NeuralNetworkCreator.git
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

## Acknowledgments

This GUI application uses TensorFlow, Keras, and PySide6.
