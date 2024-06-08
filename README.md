# ISP_Solar_Flare_Activity

This repository is for my ISP project.  It is about predicting solar flare activity. My goal is to predict the peak time of solar flares based on historical data of solar flare characteristics. This involves building a predictive model that can analyze past solar flare data and forecast the time when the next peak flare might occur.

To undertake this task I will first ensure there are no missing values that could negatively impact model performance. Convert any date-time columns to a proper datetime format for easier manipulation and feature extraction. Create new features from the datetime column such as year, month, day, hour, and minute. Identify and create relevant features that might help in predicting the peak time. This could include the physical parameters of the solar flares and any derived features from the datetime column. Divide the dataset into training and testing sets to evaluate the model's performance on unseen data. Select an appropriate machine learning model for time series prediction. Common choices include Random Forest, LSTM (Long Short-Term Memory), ARIMA (AutoRegressive Integrated Moving Average), etc. Use the training dataset to train the model. Test the model on the testing dataset. Once the model is trained and evaluated, it can be used to predict the peak times for future solar flare data. By following these steps, I aim to build a robust model capable of forecasting the peak times of solar flares, which is crucial for understanding solar activity and mitigating its effects on Earth-based technologies. The most important information in this data set are the: start date, start time, peak, duration, end time, and duration.

Dataset used: https://www.kaggle.com/datasets/khsamaha/solar-flares-rhessi 



