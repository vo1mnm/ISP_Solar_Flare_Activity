# ISP_Solar_Flare_Activity
This repository is for my ISP project.  It is about predicting solar flare activity.

My goal is to predict the peak time of solar flares based on historical data of solar flare characteristics. This involves building a predictive model that can analyze past solar flare data and forecast the time when the next peak flare might occur.

Steps Overview:

Data Preprocessing:

Handle Missing Values: Ensure there are no missing values that could negatively impact model performance.
Convert DateTime: Convert any date-time columns to a proper datetime format for easier manipulation and feature extraction.
Feature Extraction: Create new features from the datetime column such as year, month, day, hour, and minute.

Feature Engineering: Identify and create relevant features that might help in predicting the peak time. This could include the physical parameters of the solar flares and any derived features from the datetime column.
Model Preparation:

Split the Data: Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.
Choose a Model: Select an appropriate machine learning model for time series prediction. Common choices include Random Forest, LSTM (Long Short-Term Memory), ARIMA (AutoRegressive Integrated Moving Average), etc.
Model Training and Evaluation:

Train the Model: Use the training dataset to train the model.
Evaluate the Model: Test the model on the testing dataset and evaluate its performance using appropriate metrics such as Mean Absolute Error (MAE).
Prediction:

Once the model is trained and evaluated, it can be used to predict the peak times for future solar flare data.

By following these steps, I aim to build a robust model capable of forecasting the peak times of solar flares, which is crucial for understanding solar activity and mitigating its effects on Earth-based technologies.

Dataset used: https://www.kaggle.com/datasets/khsamaha/solar-flares-rhessi 

The most important information in this data set are the: start date, start time, peak, duration, end time, and duration.
