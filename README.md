📊 Mobile Price Prediction – Regression Analysis

This project demonstrates how different machine learning regression algorithms can be applied to predict the price range of mobile phones based on their features (RAM, battery power, storage, camera, etc.).

📌 Project Overview

An individual starting a new mobile company wants to estimate mobile phone prices more accurately to compete with established brands like Apple and Samsung.
Using a dataset of mobile features, this project explores the relationship between these features and mobile price ranges.

📂 Repository Contents

RegressionAnalysis_.py → Python code implementing regression models with hyperparameter tuning

Mobi_Dataset.csv → Dataset containing mobile phone features and price ranges

RegressionAnalysisReport.pdf → Detailed analysis report with findings and conclusions

🛠️ Features of the Project

Data preprocessing & correlation heatmap visualization

Regression models implemented:

Linear Regression (with & without regularization: L1, L2, ElasticNet)

Random Forest Regression

Support Vector Regression (SVR)

Hyperparameter tuning with GridSearchCV & cross-validation

Evaluation using R² Score & Adjusted R² Score

🚀 Key Insights

RAM and Battery Power are the most important predictors of mobile price.

All models performed reasonably well, but Random Forest Regressor achieved the best accuracy, capturing non-linear relationships.

⚙️ Installation & Usage

Clone the repository:

git clone https://github.com/your-username/mobile-price-regression.git
cd mobile-price-regression


Install dependencies:

pip install -r requirements.txt


Run the analysis:

python RegressionAnalysis_.py

📈 Example Output

Correlation Heatmap (via Plotly)

Best model parameters (via GridSearchCV)

R² and Adjusted R² scores for comparison

Feature importance rankings (Random Forest)

🧾 Conclusion

While Linear Regression and SVR gave decent performance, Random Forest Regressor proved to be the most reliable model for this dataset.
