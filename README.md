# House Price Prediction Web App

This project is a machine learning application that predicts house prices based on various input features using a simple linear regression model. The project includes data cleaning, model training, evaluation, and a web interface for user interaction.

## Project Overview

The goal of this project is to build a machine learning model that can predict the price of a house based on features such as the number of bedrooms, bathrooms, square footage, and more. The model is implemented using Python and trained with the `LinearRegression` algorithm from `scikit-learn`.

A web interface built with `Streamlit` allows users to input property details and receive price predictions interactively.

## Features

- Data cleaning and preprocessing
- Feature selection for model training
- Linear regression model implementation
- Model evaluation using metrics such as Mean Squared Error (MSE) and R-squared
- Interactive web app using Streamlit
- Visualizations of actual vs. predicted prices and residual distributions

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data manipulation: `pandas`, `numpy`
  - Data visualization: `matplotlib`, `seaborn`
  - Machine learning: `scikit-learn`
  - Web app: `Streamlit`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/house_price_prediction.git
   cd house_price_prediction
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the web app:

bash
Copy code
streamlit run app.py
Usage
Navigate to http://localhost:8501 in your browser to access the web app.
Enter the property details using the sidebar inputs.
View the predicted house price and analysis results.

Model Evaluation
The model was evaluated using:

Mean Squared Error (MSE): Measures the average of the squares of the errors.
R-squared: Indicates how well the model fits the data.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Datasets used open source kaggle. url: https://www.kaggle.com/datasets/itsnahm/data-house?resource=download
Special thanks to the Python and data science community for their valuable resources.
Contact
For questions or feedback, please reach out at kakani1407@gmail.com or connect on LinkedIn: https://www.linkedin.com/in/hemanth-kumar-kakani-7170341ab/
