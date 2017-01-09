# Import libraries and methods
import pandas as pd
import sqlite3
import nltk
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from nltk.stem import WordNetLemmatizer
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,accuracy_score
from math import sqrt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import svm, neighbors, ensemble


# Download corpus from NLTK library
#nltk.download('stopwords')
#nltk.download('punkt')

# Obtain 'English' stop words
stops = set(stopwords.words("english"))

# Initialize the "TfidfVectorizer" object, which is scikit-learn's bag of words and tfid tool.
vectorizer = TfidfVectorizer(max_features=None, ngram_range=(1,2))

# Define word_lemmatize to perform lemmatization
def word_lemmatize(word_lemma):
    input_lemma = []
    for word in word_lemma:
        word = WordNetLemmatizer().lemmatize(word)
        input_lemma.append(word)
    return input_lemma

# Define plot_confusion_matrix to plot confusion matrix for each model
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',  cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center", color="grey" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Format positive to 1 and negative to 0
def format(x):
    predicted_mapping = []
    for score in x:
        if score == 'negative':
            output = 0
        else:
            output = 1
        predicted_mapping.append(output)
    return predicted_mapping

# Create connection to sqllite database and import data using SQL query
connection = sqlite3.connect('database.sqlite')
dataset1 = pd.read_sql_query("""SELECT Score, Summary, Text FROM Reviews WHERE Score < 3""", connection)
dataset2 = pd.read_sql_query("""SELECT Score, Summary, Text FROM Reviews WHERE Score > 3""", connection)


# Number of rows of positive dataset
r2,c2=dataset2.shape

# Obtain one-fourth of positive dataset
dataset3=dataset2.head(math.ceil(r2/4))

# Merging positive and negative dataset
data=[dataset1,dataset3]
dataset=pd.concat(data)
r,c=dataset.shape

dataset = shuffle(dataset)

# Assign seperate variables to store specific data of the original dataset
score_data = dataset['Score']
summary_data = dataset['Summary']
text_data = dataset['Text']

# Remove puncuations from summary and text data. Keep only letters.
text = text_data.str.replace('[^a-zA-Z]'," ")
summary = summary_data.str.replace('[^a-zA-Z]'," ")

# Map the score rating to positive/negative
result = []
for score in score_data:
    if score < 3:
        output = 'negative'
    else:
        output = 'positive'
    result.append(output)

# Make every review lower case, and tokenize and lemmatize
corpus = []
for word in text:
    word = word.lower()    
    word = nltk.word_tokenize(word)
    word = word_lemmatize(word)
    #tokens = [w for w in word if not w in stops]
    corpus.append(' '.join(word))


# fit_transform() does two functions: First, it fits the model and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of strings.
prediction = dict()
test=dict()
results = []
rms = dict()
k_fold = KFold(n=len(corpus), n_folds=2)
for k in [500000]:
    scores = []

    for train_indices, test_indices in k_fold:
        x_train=[]
        x_test=[]
        y_train=[]
        y_test = []
        for i in train_indices:
            x_train.append(corpus[i])
            y_train.append(result[i])
        for j in test_indices:
            x_test.append(corpus[j])
            y_test.append(result[j])
        
        X_train_tfidf = vectorizer.fit_transform(x_train)
        X_test_tfidf = vectorizer.transform(x_test)
        
        # Chi-Square test
        print(k,"k value")
        ch2 = SelectKBest(chi2, k=k)
        X_train_tfidf = ch2.fit_transform(X_train_tfidf, y_train)
        X_test_tfidf = ch2.transform(X_test_tfidf)
        
        model = LogisticRegression(C=1e5).fit(X_train_tfidf, y_train)
        prediction['LR'] = model.predict(X_test_tfidf)
        print(metrics.classification_report(y_test, prediction['LR'], target_names = ["positive", "negative"]))
        # Compute the confusion matrix
        nb_cnf_matrix = confusion_matrix(y_test, prediction['LR'])
        np.set_printoptions(precision=10)
        test['LR']= test.get('LR',np.matrix("0 0;0 0"))+nb_cnf_matrix
        rms['LR'] = rms.get('LR',0)+mean_squared_error(format(list(y_test)),  format(prediction['LR']))
        
        model = MultinomialNB().fit(X_train_tfidf,y_train)
        # Test the model using testing data
        prediction['Multinomial'] = model.predict(X_test_tfidf)
        print(metrics.classification_report(y_test, prediction['Multinomial'], target_names = ["positive", "negative"]))
        # Compute the confusion matrix
        nb_cnf_matrix = confusion_matrix(y_test, prediction['Multinomial'])
        np.set_printoptions(precision=2)
        #test['Multinomial']= test.get('Multinomial',np.matrix("0 0;0 0"))+nb_cnf_matrix
        #rms['Multinomial'] = rms.get('Multinomial',0)+accuracy_score(format(list(y_test)),  format(prediction['Multinomial']),normalize=False)
    
        #model = svm.SVC(kernel = 'poly').fit(X=X_train_tfidf,y=y_train)
        #prediction['SVM'] = model.predict(X_test_tfidf)
        #print(metrics.classification_report(y_test, prediction['SVM'], target_names = ["positive", "negative"]))
        # Compute the confusion matrix
        #nb_cnf_matrix = confusion_matrix(y_test, prediction['SVM'])
        #np.set_printoptions(precision=2)
        #test['SVM']= test.get('SVM',np.matrix("0 0;0 0"))+nb_cnf_matrix
        
    plt.figure()
    plot_confusion_matrix(test['Multinomial'], classes=['positive','negative'],title='Multinomial Naive Bayes Confusion Matrix (Without Normalization)')
    
       
    plt.figure()
    plot_confusion_matrix(test['LR'], classes=['positive','negative'],title='LR Confusion Matrix (Without Normalization)')
    
    #plt.figure()
    #plot_confusion_matrix(test['SVM'], classes=['positive','negative'],title='LR Confusion Matrix (Without Normalization)')
        # Plot normalized confusion matrix
    print("accuracy of logistic regression",(test['LR'].item((0, 0))+test['LR'].item((1, 1)))/np.sum(test['LR']))
    print("accuracy of Multinomial",(test['Multinomial'].item((0, 0))+test['Multinomial'].item((1, 1)))/np.sum(test['Multinomial']))
    print("root mean square error of logistic regression",rms['LR']/np.sum(test['LR']))
    print("root mean square error of Multinomial",rms['Multinomial'])
plt.show()

 
# fit_transform() does two functions: First, it fits the model and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of strings.
prediction = dict()
test=dict()
results = []
rms = dict()

#Second comparision technique ROC curve    
x_train, x_test = train_test_split(corpus,test_size = 0.3, random_state=43)
y_train, y_test = train_test_split(result,test_size = 0.3, random_state=43)

vectorizer = TfidfVectorizer(max_features=None,ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)

ch2 = SelectKBest(chi2, k=2000)
X_train_tfidf = ch2.fit_transform(X_train_tfidf, y_train)
X_test_tfidf = ch2.transform(X_test_tfidf)
# Define prediction as a dictionary to hold the predicted values of each model
prediction = dict()

# Train the model using training data
model = MultinomialNB().fit(X_train_tfidf,y_train)
# Test the model using testing data
prediction['Multinomial'] = model.predict(X_test_tfidf)

# Train the model using training data
model = LogisticRegression(C=1e5).fit(X_train_tfidf, y_train)
# Test the model using testing data
prediction['LR'] = model.predict(X_test_tfidf)

# Train the model using training data
model = LinearSVC().fit(X_train_tfidf,y_train)
# Test the model using testing data
prediction['SVM'] = model.predict(X_test_tfidf)

# Train the model using training data
model = ensemble.ExtraTreesClassifier().fit(X_train_tfidf, y_train)
# Test the model using testing data
prediction['Edge Tree Classifier'] = model.predict(X_test_tfidf)

# Train the model using training data
model = ensemble.RandomForestClassifier().fit(X_train_tfidf, y_train)
# Test the model using testing data
prediction['Random Forest Classifier'] = model.predict(X_test_tfidf)

# Plot ROC Curve
y_test_mapping = []
for score in y_test:
    if score == 'negative':
        output = 0
    else:
        output = 1
    y_test_mapping.append(output)
cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(np.array(y_test_mapping), np.array(format(predicted)))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1
plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
