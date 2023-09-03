import requests
import numpy as np
import csv


def buscarReviews(itemId):
    x = requests.get(
        'https://api.mercadolibre.com/reviews/item/'+itemId+'?limit=200')
    retornoJson = x.json()
    return retornoJson['reviews']


def salvarArr(arr):
    file = open('baseNova.csv', 'a', encoding='UTF-8', newline='')
    writer = csv.writer(file, delimiter=';')
    writer.writerows(arr)
    file.close()


def carregaItens():
    arrItens = []
    with open("Dados/itens 32000.csv") as file_name:
        reader = csv.reader(file_name, delimiter=",")
        for row in reader:
            arrItens.append(row)
    return arrItens


def buscaPalavra(palavra):
    with open(r'log.txt', 'r') as file:
        content = file.read()
        if palavra in content:
            return True
        else:
            return False


arrItens = carregaItens()
arquivo_log = open('log.txt', 'a')
for linhaItens in arrItens:
    for item in linhaItens:
        if(buscaPalavra(item) == False):
            arrComentarios = np.array(buscarReviews(item))
            arquivo_log.write('\nGravando ' + str(len(arrComentarios)
                                                  )+' comentarios do item ' + item)
            print('Gravando ' + str(len(arrComentarios)) +
                  ' comentarios do item ' + item)
            arrReview = []
            for review in arrComentarios:
                conteudo = review['title'] + ' ' + \
                    review['content'].replace(";", " ").replace(
                        '\n', ' ').replace('\r', '')
                arrReview.append([
                    conteudo, review['rate']])
            salvarArr(arrReview)
arquivo_log.close()
