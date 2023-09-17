#region imports
import pandas as pd
from string import punctuation
import unidecode
import re
import numpy as np
import nltk
from nltk import tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
#endregion

#region preparando base
resenha = pd.read_csv("balanced_output.csv")
classificacao = resenha["sentiment"].replace(["neg", "pos"], [0, 1])
resenha["classificacao"] = classificacao
todas_palavras = ' '.join([texto for texto in resenha.text_pt])
token_espaco = tokenize.WhitespaceTokenizer()
token_pontuacao = tokenize.WordPunctTokenizer()
palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)
pontuacao_stopwords = pontuacao + palavras_irrelevantes
stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]
stop_words = nltk.corpus.stopwords.words('portuguese')

def remover_letras_duplicadas(frase):
    frase_sem_duplicatas = ""
    for letra in frase:
        if not frase_sem_duplicatas or letra != frase_sem_duplicatas[-1]:
            frase_sem_duplicatas += letra
    return frase_sem_duplicatas

def tratamento_unico(resenha):
  frase_processada = list()
  stemmer = nltk.RSLPStemmer()
  for opiniao in resenha.text_pt:
      nova_frase = list()
      opiniao = opiniao.lower()
      opiniao = unidecode.unidecode(opiniao)
      opiniao = re.sub(f"[{re.escape(punctuation)}]", " ", opiniao)
      opiniao.split()
      opiniao = remover_letras_duplicadas(opiniao)
      palavras_texto = token_espaco.tokenize(opiniao)
      for palavra in palavras_texto:
          if palavra not in palavras_irrelevantes and palavra not in stopwords_sem_acento:
              nova_frase.append(stemmer.stem(palavra))
      frase_processada.append(' '.join(nova_frase))
  resenha["tratamento_unico"] = frase_processada

tratamento_unico(resenha)
#endregion

#region preparando a base para analise

# unigrama CV
vect_uni_cv = CountVectorizer(ngram_range=(1,1), stop_words=stop_words)
vect_uni_cv.fit(resenha.tratamento_unico)
text_vect_uni_cv = vect_uni_cv.transform(resenha.tratamento_unico)

# separando base de treinamento x teste do Unigrama com Count Vectorizer - UCV
X_trainUCV, X_testUCV, y_trainUCV, y_testUCV = train_test_split(text_vect_uni_cv, resenha["sentiment"], test_size = 0.2, random_state = 123)

# unigrama IDF
vect_uni_idf = TfidfVectorizer(ngram_range=(1,1), use_idf=True, norm='l2', stop_words=stop_words)
vect_uni_idf.fit(resenha.tratamento_unico)
text_vect_uni_idf = vect_uni_idf.transform(resenha.tratamento_unico)

# separando base de treinamento x teste do Unigrama com Tfidf Vectorizer - UIDF
X_trainUIDF, X_testUIDF, y_trainUIDF, y_testUIDF = train_test_split(text_vect_uni_idf, resenha["sentiment"], test_size = 0.2, random_state = 123)


# bigrama CV
vect_bi_cf = CountVectorizer(ngram_range=(2,2), stop_words=stop_words)
vect_bi_cf.fit(resenha.tratamento_unico)
text_vect_bi_cf = vect_bi_cf.transform(resenha.tratamento_unico)

# separando base de treinamento x teste do Bigrama com Count Vectorizer - BCV
X_trainBCV, X_testBCV, y_trainBCV, y_testBCV = train_test_split(text_vect_bi_cf, resenha["sentiment"], test_size = 0.2, random_state = 123)

# bigrama IDF
vect_bi_idf = TfidfVectorizer(ngram_range=(2,2), use_idf=True, norm='l2', stop_words=stop_words)
vect_bi_idf.fit(resenha.tratamento_unico)
text_vect_bi_idf = vect_bi_idf.transform(resenha.tratamento_unico)

# separando base de treinamento x teste do Bigrama com Tfidf Vectorizer - BIDF
X_trainBIDF, X_testBIDF, y_trainBIDF, y_testBIDF = train_test_split(text_vect_bi_idf, resenha["sentiment"], test_size = 0.2, random_state = 123)

#endregion     

#region DecisionTree unigrama CV
treeUCV = DecisionTreeClassifier(random_state=123)
treeUCV.fit(X_trainUCV, y_trainUCV)
treeUCV.score(X_trainUCV, y_trainUCV)

y_predictionUCV = treeUCV.predict(X_testUCV)
accuracyUCV = accuracy_score(y_predictionUCV, y_testUCV)
print("Acurácia do Unigrama DecisionTree Count Vectorizer: ", accuracyUCV)

#endregion

#region unigrama DecisionTree IDF
treeUIDF = DecisionTreeClassifier(random_state=123)
treeUIDF.fit(X_trainUIDF, y_trainUIDF)
treeUIDF.score(X_trainUIDF, y_trainUIDF)

y_predictionUIDF = treeUIDF.predict(X_testUIDF)
accuracyUIDF = accuracy_score(y_predictionUIDF, y_testUIDF)
print("Acurácia do Unigrama DecisionTree Tfidf Vectorizer: ", accuracyUIDF)

#endregion

#region bigrama DecisionTree CV
treeBCV = DecisionTreeClassifier(random_state=123)
treeBCV.fit(X_trainBCV, y_trainBCV)
treeBCV.score(X_trainBCV, y_trainBCV)

y_predictionBCV = treeBCV.predict(X_testBCV)
accuracyBCV = accuracy_score(y_predictionBCV, y_testBCV)
print("Acurácia do Bigrama DecisionTree Count Vectorizer: ", accuracyBCV)

#endregion

#region bigrama DecisionTree IDF
treeBIDF = DecisionTreeClassifier(random_state=123)
treeBIDF.fit(X_trainBIDF, y_trainBIDF)
treeBIDF.score(X_trainBIDF, y_trainBIDF)

y_predictionBIDF = treeBIDF.predict(X_testBIDF)
accuracyBIDF = accuracy_score(y_predictionBIDF, y_testBIDF)
print("Acurácia do Bigrama DecisionTree Tfidf Vectorizer: ", accuracyBIDF)

#endregion

#region unigrama RandomForest CV
rfc_uni = RandomForestClassifier(bootstrap=True, criterion='gini', random_state=123)
rfc_uni.fit(X_trainUCV, y_trainUCV)

y_prediction_rfc_uni_cv = rfc_uni.predict(X_testUCV)
accuracy_rfc_uni_cv = accuracy_score(y_prediction_rfc_uni_cv, y_testUCV)
print("Acurácia do Unigrama RandomForest CV: ", accuracy_rfc_uni_cv)

#endregion

#region unigrama RandomForest IDF
rfc_uni = RandomForestClassifier(bootstrap=True, criterion='gini', random_state=123)
rfc_uni.fit(X_trainUIDF, y_trainUIDF)

y_prediction_rfc_uni = rfc_uni.predict(X_testUIDF)
accuracy_rfc_uni = accuracy_score(y_prediction_rfc_uni, y_testUIDF)
print("Acurácia do Unigrama RandomForest Tfidf: ", accuracy_rfc_uni)

#endregion

#region bigrama RandomForest CV
rfc_bi_cv = RandomForestClassifier(bootstrap=True, criterion='gini', random_state=123)
rfc_bi_cv.fit(X_trainBCV, y_trainBCV)

y_prediction_rfc_bi_cv = rfc_bi_cv.predict(X_testBCV)
accuracy_rfc_bi_cv = accuracy_score(y_prediction_rfc_bi_cv, y_testBCV)
print("Acurácia do Bigrama RandomForest CV: ", accuracy_rfc_bi_cv)

#endregion

#region bigrama RandomForest IDF
rfc_bi = RandomForestClassifier(bootstrap=True, criterion='gini', random_state=123)
rfc_bi.fit(X_trainBIDF, y_trainBIDF)

y_prediction_rfc_bi = rfc_bi.predict(X_testBIDF)
accuracy_rfc_bi = accuracy_score(y_prediction_rfc_bi, y_testBIDF)
print("Acurácia do Bigrama RandomForest Tfidf: ", accuracy_rfc_bi)

#endregion

#region regressao linear unigrama CV
lr_UCV = LogisticRegression(solver='lbfgs')
lr_UCV.fit(X_trainUCV, y_trainUCV)

y_prediction_lrUCV = lr_UCV.predict(X_testUCV)
accuracy_lrUCV = accuracy_score(y_prediction_lrUCV, y_testUCV)
print("Acurácia do Unigrama CV Regressão Logística: ", accuracy_lrUCV)

#endregion

#region regressao linear unigrama IDF
lr_UIDF = LogisticRegression(solver='lbfgs')
lr_UIDF.fit(X_trainUIDF, y_trainUIDF)

y_prediction_lrUIDF = lr_UIDF.predict(X_testUIDF)
accuracy_lrUIDF = accuracy_score(y_prediction_lrUIDF, y_testUIDF)
print("Acurácia do Unigrama Tfidf Regressão Logística: ", accuracy_lrUIDF)

#endregion

#region regressao linear bigrama CV
lr_BCV = LogisticRegression(solver='lbfgs')
lr_BCV.fit(X_trainBCV, y_trainBCV)

y_prediction_lrBCV = lr_BCV.predict(X_testBCV)
accuracy_lrBCV = accuracy_score(y_prediction_lrBCV, y_testBCV)
print("Acurácia do Bigrama CV Regressão Logística: ", accuracy_lrBCV)

#endregion

#region regressao linear bigrama IDF
lr_BIDF = LogisticRegression(solver='lbfgs')
lr_BIDF.fit(X_trainBIDF, y_trainBIDF)

y_prediction_lrBIDF = lr_BIDF.predict(X_testBIDF)
accuracy_lrBIDF = accuracy_score(y_prediction_lrBIDF, y_testBIDF)
print("Acurácia do Bigrama Tfidf Regressão Logística: ", accuracy_lrBIDF)

#endregion

#region gridSearch
# setup do grid search
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = { 'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap }
rfc = RandomForestClassifier()  
# unigrama - 3 fold CV, 10 combinations
rfc_random_uni = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rfc_random_uni.fit(X_trainUIDF, y_trainUIDF)
rcf_uni_best_params = rfc_random_uni.best_params_

rcf_uni = RandomForestClassifier(bootstrap=rcf_uni_best_params["bootstrap"], max_depth=rcf_uni_best_params["max_depth"], max_features=rcf_uni_best_params["max_features"], min_samples_leaf=rcf_uni_best_params["min_samples_leaf"], min_samples_split=rcf_uni_best_params["min_samples_split"], n_estimators=rcf_uni_best_params["n_estimators"], random_state=123)
rcf_uni.fit(X_trainUIDF, y_trainUIDF)

y_prediction_rcf_uni = rcf_uni.predict(X_testUIDF)
accuracy_rcf_uni = accuracy_score(y_prediction_rcf_uni, y_testUIDF)
print("Acurácia do Random Forest Classifier (com Grid Search) - Unigrama: ", accuracy_rcf_uni)
# bigrama - 3 fold CV, 10 combinations
rfc_random_bi = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rfc_random_bi.fit(X_trainBIDF, y_trainBIDF)
rcf_bi_best_params = rfc_random_bi.best_params_
rcf_bi = RandomForestClassifier(bootstrap=rcf_bi_best_params["bootstrap"], max_depth=rcf_bi_best_params["max_depth"], max_features=rcf_bi_best_params["max_features"], min_samples_leaf=rcf_bi_best_params["min_samples_leaf"], min_samples_split=rcf_bi_best_params["min_samples_split"], n_estimators=rcf_bi_best_params["n_estimators"], random_state=123)
rcf_bi.fit(X_trainBIDF, y_trainBIDF)

#   regressao linear com gridSearch
lr_UIDF_grid = LogisticRegression()
lr_UIDF_grid_values = { 'penalty': ['l2'], 'C': [0.001,.009,0.01,.09,1,5,10,25] }

lr_UIDF_grid_cv = GridSearchCV(lr_UIDF_grid, param_grid=lr_UIDF_grid_values, scoring='accuracy')
lr_UIDF_grid_cv.fit(X_trainUIDF, y_trainUIDF)

y_prediction_lr_grid_UIDF = lr_UIDF_grid_cv.predict(X_testUIDF)
accuracy_lr_grid_UIDF = accuracy_score(y_prediction_lr_grid_UIDF, y_testUIDF)
print("Acurácia do Unigrama Regressão Logística com Grid Search: ", accuracy_lr_grid_UIDF)

#endregion
