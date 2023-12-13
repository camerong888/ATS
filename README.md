Note: to visualize the Tree and are on Mac, you may need to run the command 'brew install graphviz'. 

Automated Trading System (ATS)
Description
The Automated Trading System (ATS) is an advanced solution designed to automate trading strategies using Python. This project focuses on leveraging machine learning algorithms for financial market predictions and employs a virtual environment for effective dependency management.

Prerequisites
Before you begin, ensure you meet the following requirements:
- Python 3.x installed on your system.
- Git (if cloning the project from a Git repository).

Installation and Setup
Setting Up the Virtual Environment
To set up the Python virtual environment and install dependencies, follow these steps:

Create Virtual Environment
1. Navigate to the project's root directory.
2. Create the virtual environment using the command: 
python -m venv venv
3. Activate the Virtual Environment:
On Windows:
venv\Scripts\activate
On Unix or MacOS:
source venv/bin/activate

Install Dependencies
Install the project dependencies from requirements.txt:
pip install -r requirements.txt

Deactivating the Virtual Environment
When you're done, deactivate the virtual environment with:

deactivate
Adding New Python Modules
If you add a new Python module, update the requirements.txt file:
pip freeze > requirements.txt

Note for Tree Visualization
To visualize the decision tree and if you are using MacOS, you may need to run the command:
brew install graphviz


Data Generation
Before running the XGBOOST Model, it's essential to generate the required data using the DataGenerator script:
python DataGenerator.py
This script must be executed to gather and prepare the data necessary for the model.

Running the XGBOOST Model
After generating the data, you can run the XGBOOST model with:
python XGBOOST_Model.py
This will start the model training and prediction process based on the generated data.