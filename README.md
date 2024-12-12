# Solubility Prediction using Machine Learning

This project predicts the solubility of chemical compounds (logS) using two machine learning models: Linear Regression and Random Forest Regressor. The dataset is sourced from [Data Professor's GitHub repository](https://github.com/dataprofessor/data).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview
This repository contains Python code to:
1. Train Linear Regression and Random Forest Regressor models on the given dataset.
2. Evaluate the models using metrics such as Mean Squared Error (MSE) and R-squared (R2).
3. Visualize the results with scatter plots comparing experimental and predicted solubility.

## Dataset
The dataset used in this project is sourced from the [Data Professor GitHub repository](https://github.com/dataprofessor/data) and contains molecular descriptors and solubility values of various compounds.

The file `delaney_solubility_with_descriptors.csv` includes the following columns:
- `logS`: The experimental solubility value (target variable).
- Other columns: Molecular descriptors (features).

## Installation
To run the program, you need to have Python installed along with the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install the required libraries using:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Ensure the dataset (`delaney_solubility_with_descriptors.csv`) is present in the repository or properly referenced in the code.

3. Run the Python script:
   ```bash
   python solubility_prediction.py
   ```

4. The program outputs the model evaluation metrics (MSE and R2) and generates a scatter plot for Linear Regression predictions.

## Results
The results from both models are summarized in a table outputted by the program. Example:

| Method            | Training MSE | Training R2 | Test MSE | Test R2 |
|-------------------|--------------|-------------|----------|---------|
| Linear Regression | 0.1234       | 0.8912      | 0.2345   | 0.8543  |
| Random Forest     | 0.0456       | 0.9423      | 0.0678   | 0.9214  |

The scatter plot shows the predicted vs. experimental solubility values for Linear Regression.

## Contributing
Contributions are welcome! If you'd like to improve this project:
1. Fork this repository.
2. Create a new branch.
3. Submit a pull request.
