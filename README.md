#Autor: Cristhian Hernández.
#Orientador: Marcos Eduardo Valle.
#Instituto de Matemática Estatística e Computação Científica.
#Universidade Estadual de Campinas.
$Campinas, São Paulo, Brasil.
#cxhernan@gmail.com.
#+593993451934.

RESUMO DO TRABALHO:
Este trabalho é a implementação em python de um algoritmo computacional que utiliza diversas técnicas de processamento digital de imagens e aprendizado de máquinas 
para contar o número de linfoblastos presentes em uma imagem de esfregaço sanguíneo obtida por microscópio com a finalidade de contribuir com o 
diagnóstico da leucemia linfoblástica aguda.

ENTRADA: Imagem colorida de esfregaço de sangue obtida por microscópio.

SAÍDA: A mesma imagem de entrada com as células analisadas pelo algoritmo marcadas como segue: se a célula foi detectada como linfoblasto, aparece um ponto vermelho nela, 
no caso contrário aparecerá um ponto verde. Além das marcações, o algoritmo mostra o número de linfoblastos detectados e a porcentagem de linfoblastos em relação ao número
de células analisadas.

INDICAÇÕES:
Para utilizar o algoritmo, é necessário rodar apenas o arquivo "main.py". Deve se considerar que na raiz tem que existir uma pasta chamada "imagenes" onde são colocadas
todas as imagens de esfregaço sanguíneo que se desejem testar no algoritmo, além disso, na variável "nome_arq_img" do arquivo main.py deve ser inicializada
com o nome do arquivo da imagem que vai se testar. As imagens utilizadas neste projeto, não foram colocadas neste repositorio, pois não são 
de nossa propriedade intelectual. Este projeto utilizou o banco de dados indicado no artigo:  ALL-IDB: THE ACUTE LYMPHOBLASTIC LEUKEMIA IMAGE DATABASE
FOR IMAGE PROCESSING.

ESTRUTURA DE AQUIVOS:
* O arquivo "main.py" contém a implementação por blocos do sistema.
* A pasta "funciones" contém vârios arquivos ".py", onde cada um têm funções que foram implementadas no main.py.
* A pasta "modelo" contém os arquivos com o modelo de rede neural convolucional utilizada e os seus pesos treinados. Esses arquivos são carregados e utilizados para predições 
  com ajuda do tensorflow.


  
