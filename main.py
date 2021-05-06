import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from funciones.func_diametro_medio import get_diametro_medio_celulas,mostrar_img
from funciones.func_segmentação import get_img_segmentada,mostrar_varias_imgs
from funciones.func_celulas_brancas import get_lista_imagenes_celulas_brancas
from funciones.func_predicciones import es_linfoblasto,get_coordenadas_scotti
from tensorflow.python.keras.models import load_model
import copy 
import  matplotlib.pyplot as plt

lista_imgs=list()
lista_titulos=list()
dir_modelo='./modelo/modelo.h5'
dir_pesos='./modelo/pesos.h5'
modelo=load_model(dir_modelo)	
modelo.load_weights(dir_pesos)
pasta_img="imagenes"
lista_arq=os.listdir(pasta_img)
nome_arq_img="Im053_1.jpg"
rota_img=pasta_img+"/"+nome_arq_img
img = cv2.imread(rota_img)
img_sin_marcas=copy.deepcopy(img)
img_scotti=copy.deepcopy(img) 
img_cinza = cv2.imread(rota_img, cv2.IMREAD_GRAYSCALE) 
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#1. Obtenção do diâmetro estimado
diametro=get_diametro_medio_celulas(img_cinza)
#2. Segmentação da imagem colorida (pegando informação das células brancas)
img_segmentada=get_img_segmentada(img_lab,diametro)
#3. Obtenção da lista de células brancas na imagem
lista_img_celulas_brancas,lista_centros_celulas_brancas=get_lista_imagenes_celulas_brancas(img_segmentada,img_sin_marcas,diametro)
cant_cel_blancas=len(lista_centros_celulas_brancas)
for c in lista_centros_celulas_brancas:
	cv2.circle(img,c, 6, (0,255,0), -1)

cant_blastos=0
cant_cel_brancas=0
i=0
#4. classificação e contagem das células que são linfoblastos
for img_celula in lista_img_celulas_brancas:
	tam_img_celula=img_celula.shape
	dim_janela=3*diametro
	if(not (0 in tam_img_celula) and tam_img_celula==(dim_janela,dim_janela,3)):
		cant_cel_brancas=cant_cel_brancas+1
		coordenadas_centro=lista_centros_celulas_brancas[i]
		if(es_linfoblasto(img_celula,modelo)):
			cv2.circle(img,coordenadas_centro, 6, (255,0,0), -1)
			cant_blastos=cant_blastos+1
	i=i+1

porc_linfoblastos=round((cant_blastos/cant_cel_brancas)*100,2)
lista_imgs.append(img_segmentada)
lista_imgs.append(img)
lista_titulos.append("")
lista_titulos.append(str(cant_blastos)+" Linfoblastos ("+str(porc_linfoblastos)+" %)")

print(nome_arq_img)
print("QUANTIDAD DE LINFOBLASTOS: "+str(cant_blastos))
mostrar_varias_imgs(lista_imgs,1,lista_titulos)



