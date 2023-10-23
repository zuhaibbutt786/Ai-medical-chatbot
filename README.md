# Common Disease Prediction using Machine Learning and NLP with Flask and KNN

## Overview
This project aims to build a web-based application that predicts common diseases based on user-provided symptoms. The system uses Machine Learning and Natural Language Processing (NLP) to process user input and K-Nearest Neighbors (KNN) for prediction. The application is built with Flask, a Python web framework, and provides a user-friendly interface for input and disease prediction.

## Features
- Predict common diseases based on user-provided symptoms.
- Utilize KNN algorithm for prediction.
- Utilize NLP for processing user input.
- Web-based user interface built with Flask.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/zuhaibbutt786/Ai-medical-chatbot
   ```

2. Change into the project directory:
   ```
   cd Ai-medical-chatbot
   ```

3. Create a virtual environment and activate it (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

4. Install required packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the Flask application:
   ```
   python app.py
   ```

6. Access the application in your web browser at `http://localhost:5000`.

## Usage
1. Open the web application in your browser.
2. Enter a list of symptoms in the input field.
3. Click the "Predict" button.
4. The application will use the KNN model to predict common diseases based on the symptoms.

## Model Training (Optional)
If you want to retrain the KNN model with your own dataset, you can follow these steps:

1. Prepare a dataset containing symptoms and corresponding disease labels.

2. Create a new Python script or Jupyter Notebook for training the KNN model. You can use libraries like scikit-learn for this purpose.

3. Train the model, save it, and update the `knn_model.pkl` file in the project directory.

4. The application will automatically load the updated model when you run it.

## Considerations
- Data Privacy: Be mindful of handling sensitive medical data and ensure compliance with data privacy regulations (e.g., HIPAA) if applicable.
- Model Accuracy: The accuracy of disease prediction depends on the quality of the training data and the features used. Continuous improvement may be needed.
- Scalability: This project is a simple demonstration. In a real-world scenario, you may need to consider scalability and system architecture.


## License
This project is licensed under the general public license License - see the(LICENSE.md) file for details.

## Author
Zuhaib Hussain Butt
