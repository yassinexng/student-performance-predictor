🎓Predicting Student GPA: A Custom Linear Regression Approach

This project demonstrates the development of a custom linear regression model to predict student Grade Point Average (GPA). By implementing the core components of the machine learning pipeline from scratch—including data preprocessing, model training, and evaluation—I gained a deeper understanding of the underlying mathematics and mechanics of linear regression and gradient descent.

Key Contributions

    End-to-End Implementation: Designed and executed a complete ML workflow, from exploratory data analysis to final model validation.

    Custom Model Development: Built the prediction function, cost calculation, and gradient descent algorithm using NumPy, avoiding high-level library abstractions for greater transparency.

    Data Preprocessing: Applied essential techniques such as missing value handling, feature selection, and standardization to ensure robust model performance.

    Visual Insights: Leveraged Matplotlib to analyze data distributions, track learning progression, and assess feature importance.

    Foundational Learning: As my first end-to-end machine learning project, this work solidified my practical understanding of core data science principles.



Dataset:

The analysis utilizes the Student Performance Dataset from Kaggle. This dataset includes various factors such as study time, absences, parental education, and extracurricular activities, which serve as features for GPA prediction.
🧠 Technologies Used

    Python: The primary programming language.

    NumPy: Essential for numerical operations and array manipulation, crucial for the custom model implementation.

    Pandas: Used for efficient data loading, manipulation, and analysis.

    Matplotlib: Employed for creating informative visualizations.

    Scikit-learn (StandardScaler): Used specifically for robust feature scaling.

How to Run This Project

To explore this project locally, follow these steps:

    Clone the Repository:

    git clone https://github.com/yassinexng/student-performance-predictor.git
    cd student-performance-predictor

    Install Dependencies:

    pip install "numpy<2" pandas matplotlib scikit-learn

    (Note: "numpy<2" is specified to ensure compatibility with other libraries compiled against NumPy 1.x. )

    Download the Dataset:

        Download the Student_performance_data .csv file from the Kaggle dataset page.

        Create a folder named data in the root directory of this repository.

        Place the downloaded Student_performance_data .csv file inside the data folder.
        Your directory structure should look like this:

    student-performance-predictor/
    ├── student_gpa_prediction.ipynb
    └── data/
        └── Student_performance_data .csv

    Launch Jupyter Notebook:

    Open Jupyter in your web browser. You can then navigate to and open student_gpa_prediction.ipynb to run the analysis.

 Feedback Welcome!

As this is my first end-to-end machine learning project, I'm genuinely keen to learn and improve. Any constructive feedback on the methodology, code, or presentation would be incredibly valuable and much appreciated. Feel free to open an issue or reach out!
