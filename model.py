import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pickle
from preprocessing import data_preprocessing, TextProcessor

def random_forest_model(train_df, text_processor, vectorizer):
    RF = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=1)
    RFpipeline = Pipeline([
        ("text_processing", text_processor),
        ("vectorizer", vectorizer),
        ("classifier", RF)
    ])
    RFpipeline.fit(train_df['Text'], train_df['Emotion'])
    return RFpipeline

def support_vector_machine_model(train_df, text_processor, vectorizer):
    svm = SVC(kernel="linear",gamma=1, C=.5, random_state=42)
    svm_pipeline = Pipeline([
        ("text_processing", text_processor),
        ("vectorizer", vectorizer),
        ("svm", svm),
    ])
    svm_pipeline.fit(train_df['Text'], train_df['Emotion'])
    return svm_pipeline

def logistic_regression_model(train_df, text_processor, vectorizer):
    logistics = LogisticRegression(random_state=42, max_iter=1000)
    logs_pipeline = Pipeline([
        ("text_processing", text_processor),
        ("vectorizer", vectorizer),
        ("classifier", logistics)
    ])
    logs_pipeline.fit(train_df['Text'], train_df['Emotion'])
    return logs_pipeline

def vote_classifier(train_df, text_processor, vectorizer):
    estimators=[
        ("RFC", random_forest_model(train_df, text_processor, vectorizer)),
        ("Logistics Regression", logistic_regression_model(train_df, text_processor, vectorizer)),
        ("SVM", support_vector_machine_model(train_df, text_processor, vectorizer))
    ]
    voting_classifier = VotingClassifier(estimators, voting='hard')
    voting_classifier.fit(train_df['Text'], train_df['Emotion'])
    return voting_classifier

def main(data_path):
    # read data
    data = pd.read_csv(data_path)

    # data preprocessing
    data = data_preprocessing(data)

    # create text processor and vectorizer
    text_processor = TextProcessor(lower=True, stem=False)
    vectorizer = CountVectorizer(max_features=3000)

    # train and save model
    model = vote_classifier(data, text_processor, vectorizer)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    main(r'E:\Full Data Science Projects\Emotions Detection NLP + ML\data.csv')