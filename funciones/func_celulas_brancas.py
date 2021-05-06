import cv2
import numpy as np
import copy
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import label

def get_lista_imagenes_celulas_brancas(img_segmentada,img,diametro_medio):
	img_bn=copy.deepcopy(img_segmentada)
	lista_img_celulas_brancas=list()
	lista_centros_celulas_brancas=list()
	contours,_ = cv2.findContours(img_segmentada, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	lista_img_celulas_brancas,lista_centros_celulas_brancas=separar_celulas(img,img_segmentada,diametro_medio,lista_img_celulas_brancas,lista_centros_celulas_brancas)
	
	return lista_img_celulas_brancas,lista_centros_celulas_brancas

def eliminar_objeto(img,img_bn,contorno):
	black = np.zeros_like(img)
	black2 = black.copy()
	cv2.drawContours(black2,[cv2.convexHull(contorno)], -1, (255, 255, 255), -1)
	g2 = cv2.cvtColor(black2, cv2.COLOR_BGR2GRAY)
	r, t2 = cv2.threshold(g2, 127, 255, cv2.THRESH_BINARY)
	img_bn=img_bn-g2
	return img_bn

def verificar_pontos_proximos(ponto_novo,lista_pontos,diametro):
	if(lista_pontos!=[]):
		k=1
		array_pontos=np.array(lista_pontos)
		vetor_distancias=np.sqrt(np.apply_along_axis(sum,1,(ponto_novo-array_pontos)**2))
		vetor_pos_pontos_com_dist_menores_ao_diametro=np.array(np.where(vetor_distancias<=k*diametro))[0]
		if(vetor_pos_pontos_com_dist_menores_ao_diametro.size<=0):
			lista_pontos_final=lista_pontos
			lista_pontos_final.append(ponto_novo)
		else:
			array_pontos[vetor_pos_pontos_com_dist_menores_ao_diametro]=(array_pontos[vetor_pos_pontos_com_dist_menores_ao_diametro]+ponto_novo)/2
			lista_pontos_final=list(map(tuple,array_pontos))
	else:
		lista_pontos_final=lista_pontos
		lista_pontos_final.append(ponto_novo)
	return lista_pontos_final


def separar_celulas(img,img_segmentada,diametro_medio,lista_img_celulas_brancas,lista_centros_celulas_brancas):
	lista_subimgs=copy.deepcopy(lista_img_celulas_brancas)
	lista_centros=copy.deepcopy(lista_centros_celulas_brancas)

	img_segmentada=img_segmentada/255
	matriz_distancias=ndimage.distance_transform_edt(img_segmentada)
	localMax=peak_local_max(matriz_distancias, indices=False, min_distance=15,labels=img_segmentada)
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-matriz_distancias, markers, mask=img_segmentada)
	j=0
	for label in np.unique(labels):
		if label == 0:
			continue
		mask = np.zeros(img_segmentada.shape, dtype="uint8")
		mask[labels == label] = 255
		contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		for i in contours:
			area_contorno = cv2.contourArea(i)
			area_casco_convexo = cv2.contourArea(cv2.convexHull(i))
			if(area_casco_convexo!=0):
				solidez=float(area_contorno)/area_casco_convexo
			else:
				solidez=0
			solidez=round(solidez,2)
			if((solidez >= 0 and solidez<=1)):
				momentos=cv2.moments(i)
				if(momentos['m00']!=0):
					cx=int(momentos['m10']/momentos['m00'])
					cy=int(momentos['m01']/momentos['m00'])
					lista_centros=verificar_pontos_proximos((cx,cy),lista_centros,diametro_medio)
				
		j=j+1
	lista_subimgs=cortar_celulas(img,lista_centros,diametro_medio)
	return lista_subimgs,lista_centros

def cortar_celulas(img,lista_centros,diametro_medio):
	lista_imgs=list()
	for cx,cy in lista_centros:
		from_col=int(cx-1.5*diametro_medio)
		to_col=int(cx+1.5*diametro_medio)
		from_fil=int(cy-1.5*diametro_medio)
		to_fil=int(cy+1.5*diametro_medio)
		lista_imgs.append(img[from_fil:to_fil,from_col:to_col])
	return lista_imgs
