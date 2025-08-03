# Student GPA Prediction: Linear Regression from Scratch

## Project Overview

This repository contains a hands-on machine learning project focused on predicting student GPAs. Unlike using pre-built libraries for the core algorithms, I've built the entire linear regression model from the ground up using fundamental Python libraries. The project walks through a complete machine learning workflow, from initial data cleaning to final model evaluation.

This project was a great way for me to get a deep understanding of the mechanics behind a machine learning model.

### Key Features:

  - **From-Scratch Implementation:** Custom Python functions for prediction, the cost function (MSE), and gradient descent.
  - **Custom Data Preprocessing:** A self-made `MyStandardScaler` class is used to standardize the data.
  - **Complete Workflow:** The project script covers data loading, cleaning, exploratory data analysis (EDA) with plots, model training, evaluation, and visualization of results.
  - **Visual Analysis:** Histograms, bar charts, and a final scatter plot are used to analyze data distributions and model performance.

## What's Included

```
student-performance-predictor/
├── Student_gpa_prediction.py    # The main Python script
├── Student_performance_data.csv   # The dataset used for this project
├── Student_performance_data.pdf   # Additional project documentation
└── README.md                      # This documentation
```
## Getting Started

To run this project, follow these simple steps.

### 1\. Clone the repository

First, get a local copy of the project files.

```bash
git clone https://github.com/yassinexng/student-performance-predictor.git
cd student-performance-predictor
```

### 2\. Install required packages

You'll need a few common data science libraries.

```bash
pip install pandas matplotlib scikit-learn
```

**Important:** You may encounter a compatibility issue with NumPy 2.x and other libraries. If you get an `ImportError` related to NumPy, please use the following command to install a compatible version:

```bash
pip install "numpy<2"
```

### 3\. Run the script

Finally, execute the Python script from your terminal.

```bash
python mainfunction.py
```

The script will print its progress to the console and generate several plots.

## Implementation Details

### Libraries Used

  - **NumPy**: For numerical operations, array manipulation, and the core linear regression algorithm implementation.
  - **Pandas**: For efficient data loading, cleaning, and manipulation of the dataset.
  - **Matplotlib**: For creating all the visualizations, including histograms, bar charts, and the final prediction vs. actual GPA plot.
  - **scikit-learn**: Used only for splitting the data into training and testing sets, keeping the core model implementation completely custom.

### Dataset

The included `Student_performance_data.csv` contains a synthetic dataset with student information, including:

  - `StudyTimeWeekly`
  - `Absences`
  - `ParentalEducation`
  - `Extracurricular` participation
  - `GPA` values (the target variable for prediction)
It can be found within the same repo of this project.
## About This Project
As my first machine learning implementation, this project represents an important step in my journey. I've focused on:

- Clear, documented code

- Thorough explanations of the math behind the algorithms

- Practical data preprocessing techniques

- Visual interpretation of results


I welcome any feedback or suggestions for improvement - please feel free to open an issue or reach out directly in my LinkedIn.
This project was my first attempt at building a complete machine learning solution from scratch. My focus was on creating clear, well-documented code that explains the fundamental principles of linear regression, from the math behind the cost function to the iterative process of gradient descent. This was a valuable learning experience, especially in overcoming common challenges like library compatibility issues. I hope this code serves as a useful and clear example for others starting their own journey in data science.
