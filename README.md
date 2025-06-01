# Remote Work Benefits: Impact on Mental Health and Productivity

## Project Overview

This project analyzes how remote work affects employee productivity, mental health, and satisfaction using a synthetic dataset of 5,000 professionals from various industries. It includes data cleaning, feature engineering, exploratory data analysis, hypothesis testing, and a machine learning model to predict remote work satisfaction.

## Objective

To explore whether remote work leads to improved productivity and mental health outcomes and to identify key factors that contribute to employee satisfaction with remote work environments.

## Dataset

* Filename: Impact\_of\_Remote\_Work\_on\_Mental\_Health.csv
* Contains 20 columns and 5,000 rows
* Includes demographic information, work location, stress levels, productivity changes, access to mental health resources, and satisfaction ratings

## Key Steps

* Data preprocessing and feature engineering
* Visualization of productivity and mental health trends
* T-test to evaluate statistical differences between remote and onsite workers
* Random forest model to predict satisfaction with remote work
* Feature importance analysis to understand key drivers

## Technologies Used

* Python
* pandas, numpy
* seaborn, matplotlib
* scikit-learn
* scipy

## Results

* Remote workers showed a higher likelihood of increased productivity and better social isolation ratings
* T-tests confirmed statistically significant differences in productivity between remote and onsite employees
* The model achieved strong classification results and highlighted work location and productivity as major predictors of satisfaction

## File Structure

* data/: contains the raw dataset
* scripts/: contains main\_analysis.py with all code
* visuals/: stores generated plots
* main.py (optional): to run the project from the root

## How to Run

1. Install dependencies:
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy

2. Run the analysis:
   `python scripts/main_analysis.py`

## License

This project is for educational and portfolio purposes.
