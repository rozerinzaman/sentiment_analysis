###########SENTIMENT ANAL WITH IMDB DATA

#from flask import Flask, render_template, url_for

# Packages for visuals
#import matplotlib as plt
import matplotlib.pyplot as plt
import nltk
import numpy as np
from sklearn.svm import SVC

import pandas as pd
import seaborn as sns;
from nltk import word_tokenize
# from sklearn import cross_validation
# from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
# alttaki 5 i veri temizleme işlemleri için
import re
import pandas as pd
from nltk.corpus import stopwords
import time

from wordcloud import WordCloud, STOPWORDS
import spacy

dff = pd.read_csv(r'C:\Users\rozalin\Desktop\tez_karısık_datasetleri\IMDB_Dataset.csv')
df= dff.sample(n=2500, random_state= 5)



#df1 = pd.DataFrame(df)


#print(df)
print("--------------dataset ------------------------------------")
print(df.info()) # DATASET HAKKINDA GENEL BİLGİLER VERİR
print("------------------ilk 5 eleman---------------------------------")
print(df.head(5)) # VERDİĞİMİZ DEĞİŞKEN SAYISI KADAR EN BAŞTAKİ ELEMENTLERİ GETİRİR
print("--------------sentiment değerleri sayıları----------------------------")
print(df["sentiment"].value_counts()) # VERDİĞİMİZ SÜTUNDAKİ DEĞELERİ SAYAR
#print("---------------------------------------------------")
#print(df[df["sentiment"] == "Positive"]) # VERDİĞİMİZ SÜTUNDAKİ DEĞERİN VERDİĞİMİZ DEĞERE EŞİT OLDUĞU SATIRLARI GETİRİR



df2 = df


"""
# hepsini küçük harf yaptık
print("tüm veriyi küçük hale getirdik\n")
df2['review'] = df2['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
print(df2.head())
print("--------------------")

# eğer yen i sütun oluşursa silmek için
#df2.drop('review',axis=1, inplace=True)
#pd.set_option('display.max_colwidth', None)
#print(df2.iloc[2])


#noktalamaları kaldırmak için
print("noktalama işaretlerinin kaldırılmış hali\n")
df2['review'] = df2['review'].str.replace('[^\w\s]','')
print(df2.head())
print("-------------------")



#nltk.download('stopwords')
#stopword leri çıkarmak için
print("stopwordlerin kaldırılmış hali\n")
stop = stopwords.words('english')
df2['review'] = df2['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(df2.head())
print("-------------------")
"""


# token lemma yrı ayrı
print("----------- token işlemi")
# token işlemi ile kelimeler parçalara ayrılır

def word_tokenize_wrapper(text):
  return word_tokenize(text)
df2['review'] = df2['review'].apply(word_tokenize_wrapper)
print(df2.head(5))



"""
#lemma
print("----------- lemma işlemi")

#lemma istemi ile kelimeler çekim eklerini kayıp eder
#lemmatization işlemi için
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('omw-1.4')
wnl = WordNetLemmatizer()


token_1 = [' '.join(review) for review in df2['review']]
#lemma istemi ile kelimeler çekim eklerini kayıp eder
#lemmatization işlemi için
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('omw-1.4')
wnl = WordNetLemmatizer()

def lemmatizee(text):
    lemmatized_tokens = [wnl.lemmatize(token) for token in text]

    return lemmatized_tokens

#lemma = lemmatizee(token_1)
#print(lemma)
df2['review'] = df2['review'].apply(lemmatizee)
print(df2.head())
"""

"""
#### tfidf ve cross validaiton

data_review= df2['review'].astype(str)
tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1,3), smooth_idf=False)
tfs = tfidf.fit_transform(data_review)
IDF_vector = tfidf.idf_
tfidf_mat = tfidf.fit_transform(data_review).toarray()


X= tfidf_mat
Y= df2['sentiment']

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
clasfc= SVC(kernel='linear')
scores = cross_val_score(clasfc, X, Y, cv=10)

print("validation:", scores.mean())
"""
#stemming kelime nmanasını bozabileceği için uygulanmaz
#df2.to_csv('dataset.csv') #dataseti 'dataset.csv2' olarak kaydetmek için kullanılır
#df2.to_excel('dataset.xlsx') #dataseti 'dataset.xlsx' olarak kaydetmek için kullanılır

df2.to_csv('dataset_deneme_bak.csv')
###### vektörizasyon ###########3
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(df2['review'], df2['sentiment'], test_size = 0.5, random_state = 0)
#burda yukarıdakiler veriler 50 50 olarak ve test train olarak sentiment ve review olarak bölnmüş
df_train90 = pd.DataFrame()
df_train90['review'] = train_X # datanın yarısı tran olarak ayrılmış review kısmı train_x sentşment kısmı train_y
df_train90['sentiment'] = train_Y

df_test10 = pd.DataFrame()
df_test10['review'] = test_X # test verileri
df_test10['sentiment'] = test_Y


#bunlar cross validate için x daha tfidf edlmedi

df_train90

df_test10

#df_train90.to_csv("data.train90")
#df_test10.to_csv("data.test10")


from sklearn.feature_extraction.text import TfidfVectorizer

#tfidf_vect_9010 = TfidfVectorizer(max_features = 35000, ngram_range=(1,1))
#chi2 kullanınca üsttekini kullan

tfidf_vect_9010 = TfidfVectorizer( preprocessor=' '.join)
#tfidf_vect_9010 = TfidfVectorizer( )


tfidf_vect_9010.fit(df2['review'])
train_X_tfidf_9010 = tfidf_vect_9010.transform(df_train90['review'])
test_X_tfidf_9010 = tfidf_vect_9010.transform(df_test10['review'])
tfidf_vect_9010
print("toke nincle")
print(train_X_tfidf_9010)



print(train_X_tfidf_9010.shape)
print(test_X_tfidf_9010.shape)
# train 6037 elemandan oluşan 2000 boyutlu vektör
# test (671, 2000)
"""
# bu işlemi ilk chi2 ye uygulamamız lazım sonra çıkanı validate için
#cross validate işlemi için
x_validate_tfidf= tfidf_vect_9010.transform(df2['review'])
y_validate= df2['sentiment']

"""
x_chi2_tfidf= tfidf_vect_9010.transform(df2['review'])
y_sentiment= df2['sentiment']

print("review boyutlar", x_chi2_tfidf.shape)
#chi2 uygulanmadan önce bakalım hangi kelimeler çoğunlukta
from sklearn.feature_selection import chi2
chi2score = chi2(x_chi2_tfidf, y_sentiment)[0]

plt.figure(figsize=(15,10))
wscores = zip(tfidf_vect_9010.get_feature_names_out(), chi2score)
wchi2 = sorted(wscores, key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
# charı göstermesi için
plt.show(block=True)
plt.interactive(False)


##chi2 feature selection
print("chi2 feture selection uygulaması")
from sklearn.feature_selection import SelectKBest, chi2
k = 2500  # Seçilecek özellik sayısı
chi2_kbest = SelectKBest(chi2, k=k)
x_kfold_chi2_tfidf = chi2_kbest.fit_transform(x_chi2_tfidf, y_sentiment)
#burdan çıkan sonuç validate e yollanır

print("feature selection dan sonra review görüntüsü\n",x_kfold_chi2_tfidf.shape)

#bunlar yukarıda tanımlandı
#x_validate= df2['review']
#y_validate= df2['sentiment']




#print(tfidf_vect_9010.vocabulary_)


# time ı hallet

start_time = time.time()
end_time = time.time()

# Süreyi hesapla
elapsed_time = end_time - start_time
"""
# cross validation with svm
from sklearn.model_selection import cross_val_score
model_svm= SVC(kernel='linear')
scores = cross_val_score(model_svm, train_X_tfidf_9010, train_Y, cv=5)
print(scores*100)
print("validation:", scores.mean()*100)
"""

from sklearn.metrics import accuracy_score, classification_report

# cross validation kfold cross_val_scor with svm
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict

kfold = KFold(n_splits=5)
model_svm = SVC(kernel='linear')

"""
# cross validaiton orginal uygulamsı
#scores = cross_val_score(model_svm, train_X_tfidf_9010, train_Y, cv=5, scoring='f1_macro')
svm_scores = cross_val_score(model_svm, x_kfold_chi2_tfidf , y_sentiment, cv=kfold)
print("valiadates", svm_scores*100)
print("standar deviation: ",  svm_scores.std()*100)
print("cross_val_score with svm validation acc : ", svm_scores.mean()*100)
#print("svm validation acc : %2.2f" % scores.mean())
"""
""""
score_pred = cross_val_predict(model_svm, x_kfold_chi2_tfidf, y_sentiment, cv=kfold)
report = classification_report(y_sentiment, score_pred)
print("yeni rapor\n", report)
"""

def cross_validation(model_adı, model, X, y, cv):
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')*100
    end_time = time.time()

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    elapsed_time = end_time - start_time

    print( model_adı, " scores: ", scores)
    print("chi2 and validate accuracy of ", model_adı, ": ", mean_accuracy)
    print(model_adı," geçen zaman: ", elapsed_time)


    score_pred = cross_val_predict(model, X, y, cv=cv)
    report = classification_report(y, score_pred)
    print(model_adı, " rapor\n", report)

    print("confusion matrix:")
    print(metrics.confusion_matrix(y, score_pred))
    print('------------------------------')





"""
#### orjinal model uygulama uzun kod
# normal svm
model_svm_orj = SVC(kernel='linear')
model_svm_orj.fit(train_X_tfidf_9010,train_Y)
# burda modele vector hale gelmiş train review verisi ve normal halde train sentiment verisi veriliyor model öğreniyor

predictions_SVM_9010 = model_svm_orj.predict(test_X_tfidf_9010)
#burda test edebilmek için SVM modeline vector hale gelmiş test review verisi veriliyor

test_prediction_9010 = pd.DataFrame() # bos bir data set kurulur
test_prediction_9010['review'] = test_X # boş datada review sütunu oluşturulup orjinal datadaki test kısmındaki review ler aktarılıyor
test_prediction_9010['sentiment'] = predictions_SVM_9010 #sentiment bloğuna önceden modele predict edilmiş vektörel test review eklenir
SVM_accuracy_9010 = accuracy_score(predictions_SVM_9010, test_Y)*100
SVM_accuracy_9010 = round(SVM_accuracy_9010,1)

#print("test_prediction\n",test_prediction_9010)
print("svm orj acc:",SVM_accuracy_9010)
print("********************")

print(metrics.classification_report(test_Y, predictions_SVM_9010, target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(test_Y, predictions_SVM_9010))
print('------------------------------')

"""

def orjinal_acc(model_adı, model,train_X_tfidf_9010, train_Y, test_X_tfidf_9010, test_X, test_Y ):



    start_time = time.time()
    model.fit(train_X_tfidf_9010, train_Y)
    # burda modele vector hale gelmiş train review verisi ve normal halde train sentiment verisi veriliyor model öğreniyor
    predictions= model.predict(test_X_tfidf_9010)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # burda test edebilmek için SVM modeline vector hale gelmiş test review verisi veriliyor
    test_prediction_9010 = pd.DataFrame()  # bos bir data set kurulur
    test_prediction_9010['review'] = test_X  # boş datada review sütunu oluşturulup orjinal datadaki test kısmındaki review ler aktarılıyor
    test_prediction_9010['sentiment'] = predictions  # sentiment bloğuna önceden modele predict edilmiş vektörel test review eklenir
    accuracy= accuracy_score(predictions, test_Y) * 100
    accuracy = round(accuracy, 1)
    # print("test_prediction\n",test_prediction_9010)
    print(model_adı,"orj acc:", accuracy)
    print(model_adı," orjj geçen zaman: ", elapsed_time)
    print("********************")
    """
    ####### rapor istersen önce bunu düzenle
    print(metrics.classification_report(test_Y, predictions, target_names=['Positive', 'Negative']))
    print("confusion matrix:")
    print(metrics.confusion_matrix(test_Y, predictions))
    print('------------------------------')
    """

from sklearn.naive_bayes import MultinomialNB
model_mnb = MultinomialNB()

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()

""""
# cross val witk kfold cross_validate with svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
scoring = {'prec_macro': 'precision_macro','rec_macro': make_scorer(recall_score, average='macro')}
scores_2 = cross_validate(model_svm,train_X_tfidf_9010, train_Y, scoring=scoring,cv=5, return_train_score=True)
sorted(scores_2.keys())
['test_prec_macro', 'test_rec_macro',
 'train_prec_macro', 'train_rec_macro']
print("cross_validate svm ", scores_2['test_prec_macro'].mean())
"""


# valiadaiton işlemleri

modelad = "SVM"
cross_validation(modelad,model_svm,x_kfold_chi2_tfidf, y_sentiment, cv=kfold)
orj_ad = "SVM"
orjinal_acc(orj_ad,model_svm,train_X_tfidf_9010, train_Y, test_X_tfidf_9010, test_X, test_Y)

modelad = "MNB"
cross_validation(modelad, model_mnb, x_kfold_chi2_tfidf, y_sentiment, cv=kfold)
orj_ad = "MNB"
orjinal_acc(orj_ad,model_mnb,train_X_tfidf_9010, train_Y, test_X_tfidf_9010, test_X, test_Y)

modelad = "DT"
cross_validation(modelad, model_dt, x_kfold_chi2_tfidf, y_sentiment, cv=kfold)
orj_ad = "DT"
orjinal_acc(orj_ad,model_dt,train_X_tfidf_9010, train_Y, test_X_tfidf_9010, test_X, test_Y)



#cross_validation(modelad, model_knn, x_kfold_chi2_tfidf, y_sentiment, cv=kfold)

#orj_ad= "KNN"
#orjinal_acc(orj_ad,model_knn,train_X_tfidf_9010, train_Y, test_X_tfidf_9010, test_X, test_Y )

# knn cros validation ve orjj knn çağırma
k_value = 1
k_number = [ 20, 25, 35, 40, 55, 60 ]
i = 0
print("---------------------------------------------")
for number in k_number:
    model_k = KNeighborsClassifier(n_neighbors=number)
    print(k_number[i]," sayısı için")
    cross_validation("KNN", model_k, x_kfold_chi2_tfidf, y_sentiment, cv=kfold)

    orjinal_acc("KNN", model_k, train_X_tfidf_9010, train_Y, test_X_tfidf_9010, test_X, test_Y)
    i = i+1
    print("----------------------------------------")

"""
#zaman için hesaplama
start_time = time.time()
end_time = time.time()
elapsed_time = end_time - start_time
"""