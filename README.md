# Analisador de Sentimentos - Projeto

## Descrição
Este projeto tem como objetivo criar um analisador de sentimentos a partir de uma base de dados composta por um conjunto de frases e notas associadas. O processo é dividido em três arquivos com funções específicas, visando formatar a base de entrada de dados e realizar o pré-processamento das frases, culminando na análise de sentimentos.

### Arquivos do Projeto
#### 1. buscaItens
O primeiro passo envolve a aquisição dos códigos de itens (produtos) que conterão as avaliações. Para isso, foi pré-adquirida uma base de categorias do site do Mercado Livre. Em seguida, uma API disponibilizada pelo Mercado Livre foi utilizada para obter os códigos dos itens pertencentes a uma categoria específica. Esses dados foram obtidos no formato JSON e salvos em um arquivo. No total, 32.000 itens foram adquiridos dessa maneira.

#### 2. buscaComentarios
Com a base de códigos dos itens em mãos, foi possível adquirir os comentários associados a esses itens usando outra API do Mercado Livre. Os IDs dos itens foram utilizados para adquirir um total de 62.000 comentários, eles foram salvos no seguinte formato:
{Titulo da avaliação}{avaliação};{nota}
Por Exemplo: Excelente Bom, pelo preço o produto é muito bom.;5

**Nota Importante:** É importante destacar que a API do Mercado Livre utilizada para essa aquisição de dados pode não estar mais disponível para uso público da mesma maneira como descrito nestes dois passos. Para repetir esta análise de sentimento, os dados devem ser adquiridos de outra forma ou de outra fonte.


#### 3. convertLine
No primeiro arquivo, chamado `convertLine`, o objetivo é transformar a base de dados de um formato para outro. Isso é feito atribuindo um "id" a cada frase e formatando as propriedades com aspas. O resultado é salvo em um novo arquivo para preservar a base de dados original. Essa etapa simplifica a leitura e escrita para os próximos passos.

#### 4. balance
O segundo arquivo, chamado `balance`, realiza o balanceamento das notas positivas e negativas da base de dados convertida. Isso garante que, na análise subsequente, a quantidade de análises positivas e negativas seja igual, evitando viés nos resultados.

#### 5. main
O terceiro arquivo, chamado `main`, é o núcleo do algoritmo. Ele recebe a base de dados, realiza o pré-processamento utilizando técnicas de processamento de linguagem natural e, em seguida, conduz a análise de sentimentos. A cada etapa de pré-processamento, uma análise de sentimento é realizada, revelando a acurácia da análise após cada tratamento.

### Técnica de Análise de Sentimento
A técnica utilizada para a análise de sentimento é a "regressão logística", um método de aprendizado de máquina que é comumente usado para tarefas de classificação.

### Técnicas de Processamento de Linguagem Natural
Foram aplicadas as seguintes técnicas de processamento de linguagem natural para o pré-tratamento do conjunto de dados:

- **Tratamento 1:** Exclusão das "stopwords".
- **Tratamento 2:** Exclusão das pontuações.
- **Tratamento 3:** Exclusão dos acentos das palavras.
- **Tratamento 4:** Padronização de todas as palavras para minúsculas.
- **Tratamento 5:** "Stemização" das frases.

A cada etapa de tratamento, a análise de sentimento é refeita para avaliar o impacto dessas transformações.

### Tecnologias Utilizadas
Este projeto foi construído utilizando Python e as seguintes bibliotecas:

- **pandas:** Para manipulação de dados e carregamento da base de dados no formato CSV.
- **scikit-learn:** Para a implementação da regressão logística e a divisão dos dados em treinamento e teste.
- **nltk:** Biblioteca de processamento de linguagem natural para tokenização.
- **seaborn e matplotlib:** Para a criação de visualizações gráficas.
- **string e unidecode:** Para o tratamento de caracteres especiais.
- **CountVectorizer e TfidfVectorizer:** Para a vetorização do texto.

### Uso
Para utilizar este projeto, siga os passos abaixo:

1. **Preparação da Base de Dados:** Certifique-se de ter uma base de dados composta por frases e notas no formato CSV.

2. **Execução dos Arquivos:** Execute os arquivos na seguinte ordem: `convertLine`, `balance`, e por fim, `main`. Isso garantirá que a base de dados seja convertida, balanceada e, em seguida, analisada.

3. **Avaliação dos Resultados:** Após a execução do `main`, você poderá avaliar a acurácia da análise de sentimento e os efeitos de cada etapa de pré-processamento.