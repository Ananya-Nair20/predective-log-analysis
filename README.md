# Predictive Security Log Analysis for Insider Threat Detection

## Overview

This project uses machine learning techniques, specifically the Isolation Forest algorithm, to detect potential insider threats by analyzing user activity logs. The goal is to identify anomalous patterns of behavior that deviate significantly from the norm, which could indicate malicious intent or compromised accounts.

The project encompasses data cleaning, feature engineering, model training, hyperparameter tuning (implicitly through validation), and anomaly detection. By extracting meaningful features from raw security logs and employing an unsupervised anomaly detection algorithm, this project aims to provide an automated and scalable solution for identifying suspicious insider activities.

## Key Features

* **Data Cleaning and Preprocessing:** The project includes steps to clean raw security log data, handling missing values and inconsistencies to ensure data quality for subsequent analysis.
* **Feature Engineering:** New, potentially more informative features are engineered from the existing log data. These features aim to capture different aspects of user behavior, such as:
    * Activity count per user within a specific time window.
    * The number of unique resources (e.g., PCs) accessed by a user.
    * The depth of file paths accessed.
    * The frequency of user activities on a daily basis.
    * Temporal features like the day of the week and hour of the day of activity.
* **Anomaly Detection using Isolation Forest:** The core of the project utilizes the Isolation Forest algorithm, an efficient unsupervised learning method particularly well-suited for anomaly detection in high-dimensional datasets.
* **Data Splitting:** The dataset is divided into training, validation, and testing sets to ensure robust model evaluation and prevent overfitting. The validation set is used to guide the selection of an appropriate anomaly score threshold.
* **Anomaly Scoring and Thresholding:** The trained Isolation Forest model assigns an anomaly score to each data point. A threshold is determined based on the validation set's anomaly scores to classify activities as either normal or potentially anomalous.
* **Evaluation:** The project includes an evaluation of the model's performance on the test set using metrics such as accuracy (although in a real-world anomaly detection scenario with imbalanced data, other metrics like precision, recall, and F1-score would be more informative).

## Project Structure: 

.
├── data/
│   └── device.csv          # Raw security log data (example)
├── cleaned/
│   └── device_clean.csv    # Cleaned security log data (output of cleaning)
├── notebooks/
│   └── Predictive_Security_Log_Analysis_for_Insider_Threat_Detection (1).ipynb
├── README.md
└── ... (other potential files like requirements.txt, etc.)

* **`data/`:** Contains the raw security log data.
* **`cleaned/`:** Stores the processed and cleaned log data.
* **`notebooks/`:** Includes the Colab Notebook (`Predictive_Security_Log_Analysis_for_Insider_Threat_Detection (1).ipynb`) containing the project's code and analysis.
* **`README.md`:** This file provides an overview of the project.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Set up the environment:** (Likely requires Python and specific libraries)
    ```bash
    pip install -r requirements.txt  # If a requirements.txt file exists
    # Or install necessary libraries manually:
    pip install pandas pyspark scikit-learn numpy
    ```

3.  **Place your data:** Ensure your security log data (e.g., `device.csv`) is in the `data/` directory.

4.  **Run the notebook:** Execute the Colab Notebook (`Predictive_Security_Log_Analysis_for_Insider_Threat_Detection (1).ipynb`) to perform data cleaning, feature engineering, model training, and anomaly detection.

## Usage

The Colab Notebook provides a step-by-step guide to the project's workflow. By running the cells in the notebook, you will:

* Load and clean the security log data.
* Engineer relevant features.
* Split the data into training, validation, and testing sets.
* Train an Isolation Forest model on the training data.
* Determine an anomaly score threshold using the validation data.
* Evaluate the model's performance on the test data.
* (Potentially) Visualize the detected anomalies.

The output will include the trained Isolation Forest model and an evaluation of its performance in identifying potential insider threats.

## Potential Future Enhancements

* **Integration with Real-time Data Streams:** Adapt the project to process and analyze security logs in real-time for proactive threat detection.
* **Advanced Feature Engineering:** Explore more sophisticated feature engineering techniques to capture subtle patterns of malicious behavior.
* **Comparison with Other Anomaly Detection Algorithms:** Evaluate the performance of other anomaly detection algorithms (e.g., One-Class SVM, Local Outlier Factor) and compare their effectiveness.
* **Incorporating Domain Expertise:** Integrate insights from security experts to refine feature selection and anomaly interpretation.
* **Visualization and Reporting:** Develop robust visualization tools and reports to effectively communicate detected anomalies and potential threats.
* **Hyperparameter Optimization:** Implement more formal hyperparameter tuning techniques (e.g., GridSearchCV, RandomizedSearchCV) to optimize the Isolation Forest model's parameters.
* **Handling Imbalanced Data:** Employ techniques to address the potential imbalance between normal and anomalous activities in the dataset.


## Acknowledgements

Data Repository URL: https://www.kaggle.com/datasets/mrajaxnp/cert-insider-threat-detection-research?resource=download
