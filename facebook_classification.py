import pandas as pd
import os
import numpy as np
import re
from datetime import datetime
from nltk.stem import SnowballStemmer

from sklearn import metrics
from sklearn.preprocessing import Imputer, LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def preprocessor(text):
    processed = re.sub(r'[#|\!|\-|\+|:|//|\']', "", text)
    processed = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', ' ', processed).strip()
    processed = re.sub('[\s]+', ' ', processed).strip()
    processed = " ".join([SnowballStemmer("english").stem(word)
                          for word in processed.split()])
    return processed


def extract_feature(train_idx, test_idx, feature_df):
    X_train = feature_df.iloc[train_idx]
    X_test = feature_df.iloc[test_idx]

    # tf-idf
    description_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                             ngram_range=(1, 2),
                                             preprocessor=preprocessor)
    description_vectorizer.fit(X_train.description.values.astype('U'))
    tfidf_train = description_vectorizer.transform(
        X_train.description.values.astype('U'))
    tfidf_test = description_vectorizer.transform(
        X_test.description.values.astype('U'))

    # LSA
    tfidf_lsa = TruncatedSVD(n_components=150, random_state=0)
    reduced_tfidf_train = tfidf_lsa.fit_transform(tfidf_train)
    reduced_tfidf_test = tfidf_lsa.transform(tfidf_test)

    # combine text and non-text features
    feature_train = np.hstack((X_train.drop('description', axis=1),
                              reduced_tfidf_train))
    feature_test = np.hstack((X_test.drop('description', axis=1),
                             reduced_tfidf_test))

    # scale features
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(feature_train)
    feature_train = feature_scaler.transform(feature_train)
    feature_test = feature_scaler.transform(feature_test)

    return feature_train, feature_test


def validate_model(clf_model, features, label):
    """
        clf_model- model object
        feaures- original feature dataset
        label- original label dataset
    """
    model_result = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    i = 0
    for train, test in kf.split(features.values):
        i = i + 1
        print ('Fold {}'.format(i))
        train_final, test_final = extract_feature(train, test, features)
        clf_model.fit(train_final, label[train])
        output = clf_model.score(test_final, label[test])
        model_result.append(output)
        # accuracy of individual folds
        print ('Accuracy:{}'.format(output))

    print ('Overall Accuracy: {}'.format(np.mean(model_result)))


# Edit the path to your data if necessary
data_path = 'Data&Data Classification Challenge - Facebook - Training Set.csv'

# load the training data
data = pd.read_csv(data_path, delimiter="\t")

# separate target label from feature data which the model will be trained on
label = data['INDEX New']
features = data.drop('INDEX New', axis=1)

features['published_hour'] = features.published_at.apply(
    lambda x: datetime.strptime(x, '%m/%d/%y %I:%M %p').hour
    if len(x) > 10 else np.nan
)

features['description_length'] = features.description.apply(
    lambda x: len(x.split()) if isinstance(x, str) else 0
)

# add 1 to differentiate NaN from having no lables
features['picture_label_occurrences'] = features.picture_labels.apply(
    lambda x: x.count(',') + 1 if (not isinstance(x, float)) or
                                  (isinstance(x, float) and not np.isnan(x))
    else 0
)

features['hashtags'] = features.description.apply(
    lambda x: x.count('#') if (not isinstance(x, float)) or
                              (isinstance(x, float) and not np.isnan(x))
    else 0
)

features['punctuations'] = features.description.apply(
    lambda x: x.count('!') if (not isinstance(x, float)) or
                              (isinstance(x, float) and not np.isnan(x))
    else 0
)

phone = re.compile(r'\s[[0-9]{9,10}|[0-9]{3,4}\s[0-9]{6,7}]\s')
contact_flag = ['call', 'contact', '@', 'whatsapp',
                'text', 'message', 'pm', phone]
features['has_contact'] = features.description.apply(
    lambda x: int(any(bool(re.search(s, re.sub(r'[\-|\+|\(|\)|\.|\,]',
                                               '', x.lower())))
                      for s in contact_flag))
    if (not isinstance(x, float)) or (isinstance(x, float) and not np.isnan(x))
    else 0
)

# add 1 to differentiate NaN from having no uppercase letters
features['uppercase_count'] = features.description.apply(
    lambda x: (sum(1 for c in x if c.isupper())) + 1
    if not isinstance(x, float)
    else 0
)

# add 1 to differentiate NaN from having no uppercase letters
features['uppercase_ratio'] = features.description.apply(
    lambda x: (sum(1 for c in x if c.isupper()) + 1) / len(x)
    if not isinstance(x, float)
    else 0
)

features['has_pic_url'] = features.pictures_url.apply(
    lambda x: 0 if not isinstance(x, float) else 1
)

# remove profile_picture, pictures_url, and
# published_at since they are not informative
features = features.drop('profile_picture', axis=1) \
    .drop('pictures_url', axis=1) \
    .drop('published_at', axis=1)

# impute nan values in published_hour
hour_imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
hour_imp.fit(features.published_hour.values.reshape(1, -1))
features.published_hour = sum(hour_imp.transform(
    features.published_hour.values.reshape(1, -1)).tolist(), [])

# one hot encode categorial variable
encoded_owner = pd.get_dummies(features.owner_type)
features = pd.concat([features, encoded_owner], axis=1)

# drop uninformative features
features = features.drop('owner_type', axis=1) \
    .drop('found_keywords', axis=1) \
    .drop('picture_labels', axis=1)

# change label categories from string to integer
LE = LabelEncoder()
label = LE.fit_transform(label)

# modeling; random forest
print ('Testing Random Forest')
clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                             min_samples_split=2, random_state=0)
validate_model(clf, features, label)

# modeling; gradient boosting
print ('Testing Gradient Boosting')
clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                 subsample=0.8, max_depth=10,
                                 max_features='auto', random_state=0)
validate_model(clf, features, label)

# modeling; adaboosting
print ('Testing AdaBoosting')
clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=5),
                         n_estimators=300, learning_rate=0.1, random_state=0)
validate_model(clf, features, label)

# modling; xgboost
# list to store results
print ('Testing XGBoost')
model_result = []
predictions = []
actuals = []

kf = KFold(n_splits=5, shuffle=True, random_state=0)
i = 0
for train, test in kf.split(features.values):
    i = i + 1
    print ('Fold {}'.format(i))
    train_final, test_final = extract_feature(train, test, features)
    actuals = actuals + label[test].tolist()

    # prepare the dataset for xgboost
    dtrain = xgb.DMatrix(train_final, label=label[train])
    dtest = xgb.DMatrix(test_final, label=label[test])
    param = {'max_depth': 7, 'eta': 0.1, 'objective': 'multi:softmax',
             'num_class': 3, 'subsample': 0.8, 'seed': 0}
    num_round = 300
    clf_xgb = xgb.train(param, dtrain, num_round)
    # make prediction
    preds = clf_xgb.predict(dtest)
    predictions = predictions + preds.tolist()

    output = metrics.accuracy_score(label[test], preds)
    model_result.append(output)
    # accuracy of individual folds
    print ('Accuracy:{}'.format(output))

print ('Overall Accuracy: {}'.format(np.mean(model_result)))
