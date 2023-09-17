import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import tokenize
import seaborn as sns
from string import punctuation
import unidecode
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              texto[coluna_classificacao],
                                                              random_state = 42)
    regressao_logistica = LogisticRegression(solver='lbfgs')
    regressao_logistica.fit(treino, classe_treino)

    nova_frase_vetorizada = vetorizar.transform(['Razoável Preço bom. No entanto o sabor deixa a desejar apesar de ser um uísque bem conceituado.'])
    resultado = regressao_logistica.predict(nova_frase_vetorizada)
    print(resultado[0])
    
    return regressao_logistica.score(teste, classe_teste)

def pareto(texto, coluna_texto, quantidade):
    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])
    frequencia = nltk.FreqDist(token_espaco.tokenize(todas_palavras))
    df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                 "Frequência": list(frequencia.values())})
    df_frequencia = df_frequencia.nlargest(columns = "Frequência", n = quantidade)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequencia, x= "Palavra", y = "Frequência", color = 'gray')
    ax.set(ylabel = "Contagem")
    plt.show() 


resenha = pd.read_csv("balanced_output.csv")

classificacao = resenha["sentiment"].replace(["neg", "pos"], [0, 1])
resenha["classificacao"] = classificacao

todas_palavras = ' '.join([texto for texto in resenha.text_pt])

palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")

frase_processada = list()
token_espaco = tokenize.WhitespaceTokenizer()
for opiniao in resenha.text_pt:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
resenha["tratamento_1"] = frase_processada

pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)
token_pontuacao = tokenize.WordPunctTokenizer()
pontuacao_stopwords = pontuacao + palavras_irrelevantes
frase_processada = list()
for opiniao in resenha["tratamento_1"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
resenha["tratamento_2"] = frase_processada

sem_acentos = [unidecode.unidecode(texto) for texto in resenha["tratamento_2"]]
stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]

resenha["tratamento_3"] = sem_acentos

frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))
    
resenha["tratamento_3"] = frase_processada

frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    opiniao = opiniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_4"] = frase_processada

stemmer = nltk.RSLPStemmer()
frase_processada = list()
for opiniao in resenha["tratamento_4"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))
    
resenha["tratamento_5"] = frase_processada


# tfidf = TfidfVectorizer(lowercase=False, max_features=50)


# tfidf_bruto = tfidf.fit_transform(resenha["tratamento_5"])
# treino, teste, classe_treino, classe_teste = train_test_split(tfidf_bruto, resenha["classificacao"], random_state = 42)

# regressao_logistica = LogisticRegression(solver='lbfgs')
# regressao_logistica.fit(treino, classe_treino)
# acuracia_tfidf_bruto = regressao_logistica.score(teste, classe_teste)


print(classificar_texto(resenha, "text_pt", "classificacao"))
print(classificar_texto(resenha, "tratamento_1", "classificacao"))
print(classificar_texto(resenha, "tratamento_2", "classificacao"))
print(classificar_texto(resenha, "tratamento_3", "classificacao"))
print(classificar_texto(resenha, "tratamento_4", "classificacao"))
print(classificar_texto(resenha, "tratamento_5", "classificacao"))
# print(acuracia_tfidf_bruto)
