import requests
import numpy as np
import csv


def buscarItens(categoriaId, offset):
    x = requests.get(
        'https://api.mercadolibre.com/sites/MLB/search?category='+categoriaId+'&limit=50&offset='+str(offset))
    retornoJson = x.json()
    return retornoJson['results']


def salvarArr(arr):
    file = open('itens.csv', 'a', encoding='UTF-8', newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow([])
    writer.writerows(arr)
    file.close()


def carregaCategorias():
    arrCategorias = []
    with open("Dados/categorias.csv") as file_name:
        reader = csv.reader(file_name, delimiter=",")
        for row in reader:
            arrCategorias.append(row)
    return arrCategorias


arrCategorias = carregaCategorias()
for linha_categoria in arrCategorias:
    for categoria in linha_categoria:
        for offset in range(0, 1000, 50):
            print('Salvando produtos da categoria ' +
                  categoria + ' offset= '+str(offset))
            arrItens = np.array(buscarItens(categoria, offset))
            arrReview = []
            for item in arrItens:
                arrReview.append([item['id']])
            salvarArr(arrReview)
