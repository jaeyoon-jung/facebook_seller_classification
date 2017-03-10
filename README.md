# ML Model to Classify Facebook Seller


This project requires Python 3.6 and the following Python libraries installed:

* pandas
* numpy
* re
* datetime
* nltk
* sklearn
* xgboost (http://xgboost.readthedocs.io/en/latest/build.html)
  
I also recommend running the code on iPython Notebook (https://jupyter.readthedocs.io/en/latest/install.html#new-to-python-and-jupyter)

## Objective

Using 'Data&Data Classification Challenge - Facebook - Training Set.csv', build a machine learning model that can classify Facebook sales posts to Fake Seller, No Seller, and Seller.

## Data Preprocessing
Original Features:
* found_keywords_occurrences	
* nb_likes			
* nb_share
* owner_type

Meta Features:
* published_hour 
* description_length
* picture_label_occurences
* hashtags 
* punctuations 
* has_contact
* uppercase_count
* uppercase_ratio
* has_pic_url 

Natural Language Processing:
* semantic features using tf-idf, reduced with SVD

## Models:

Models are validated with K-Fold cross validation with 5 folds.

* Random Forest: 0.850 accuracy
* Gradient Boosting with Decision Tree Regressor: 0.857 accuracy
* AdaBoost: 0.837 accuracy
* eXtreme Gradient Boosting (XGBoost): 0.860 accuracy

Although all 4 of them are very close to one another in terms of accuracy, XGBoost had the best performance, both accuracy and run-time wise. 

## XGBoost Performance:

| label | precision | recall | f1-score | 
| ----- | --------- | ------ | ---------|
| Fake Seller | 0.87 | 0.81 |  0.84 | 
|  No Seller | 0.87 | 0.90 | 0.89 |
| Reseller | 0.83 | 0.84 | 0.83 |
| total |  0.86 |  0.86 |  0.86 |
