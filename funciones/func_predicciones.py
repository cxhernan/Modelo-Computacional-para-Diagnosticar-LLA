import numpy as np
import cv2

def es_linfoblasto(img,modelo):
	img=cv2.resize(img,(100,100))
	img=img/255
	img=np.array(img)
	Xnew=np.expand_dims(img,axis=0)
	Ypred = modelo.predict(Xnew,verbose=0)
	Ypred=np.round(Ypred,0)
	prediccion=Ypred[0][0]
	if(prediccion==1):
		return True
	else:
		return False

def get_coordenadas_scotti(nome_arq_coord_scotti):
	coordenadas_scotti=list()
	indice=nome_arq_coord_scotti.find(".")
	nome_arq_coord_scotti=nome_arq_coord_scotti[:indice]
	nome_arq_coord_scotti=nome_arq_coord_scotti.strip()
	pasta_coord_res_scotti="xyc"
	
	nome_arq_coord_scotti=nome_arq_coord_scotti+".xyc"
	rota_coord_scotti=pasta_coord_res_scotti+"/"+nome_arq_coord_scotti
	arq=open(rota_coord_scotti,"r")
	lista_txt=arq.read().split()
	for i in range(0,int(len(lista_txt)/2)):
		j=2*i
		coord=(int(lista_txt[j]),int(lista_txt[j+1]))
		coordenadas_scotti.append(coord)
	return coordenadas_scotti