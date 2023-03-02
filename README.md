# NLP-hotel-reviews
Final exam project for Data Science Lab course at PoliTO

This script performs sentiment analysis on Italian texts, specifically reviews of hotels. It first loads a dataset containing reviews labeled as positive or negative and performs data cleaning, including tokenization and stemming. The cleaned data is then split into training and test sets, and a supervised learning model is trained on the training set using a TfidfVectorizer for feature extraction and a LinearSVC as the classification model. The model is evaluated on the test set, and the results are printed, including classification report and f1-score. Additionally, a confusion matrix and a plot of the most common terms for positive and negative reviews are created. Finally, the model is used to predict the sentiment of a separate set of reviews and output to a CSV file.

