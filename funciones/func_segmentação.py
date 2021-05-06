import cv2
import numpy as np
import math
from itertools import groupby
import operator
from scipy import ndimage
import copy
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage.morphology import area_opening

def get_img_segmentada(img,diametro_medio):
	#gerar_nuvem_pontos(a,b)
	img_ref=gerar_img_ref(img)
	img_a,img_b,img_c=agrupar(img)
	img_escolhida=escolher_imagen(img_ref,img_a,img_b,img_c)

	img_limpa =np.uint8(255 * ndimage.binary_fill_holes(img_escolhida).astype(int))
	#img_limpa=img_escolhida
	#k=0.1
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(round(k*diametro_medio,0)),int(round(k*diametro_medio,0))))
	#img_limpa = cv2.morphologyEx(img_limpa, cv2.MORPH_OPEN, kernel)
	


	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(round(diametro_medio,0)), int(round(diametro_medio,0))))
	#img_open_normal = cv2.morphologyEx(img_limpa, cv2.MORPH_OPEN, kernel)
	area_limear=sum(sum(kernel))
	#print(kernel)
	#print(sum(sum(kernel)))
	img_segmentada=area_opening(img_limpa,area_threshold=area_limear,connectivity=1,parent=None,tree_traverser=None)

	return img_segmentada


def gerar_nuvem_pontos(img_a,img_b):
	plt.figure(figsize=(10,7))
	x = np.array(img_a).flatten()
	y = np.array(img_b).flatten()
	plt.plot(x,y,'o',markersize=2)
	plt.show()

def gerar_img_ref(img):
	L,a,b=cv2.split(img)
	img_vetor_ord = np.sort(np.array(b).flatten())
	p=list(set(img_vetor_ord))
	f=[len(list(group)) for key, group in groupby(img_vetor_ord)]
	pontos_histog=np.array(list(zip(p,f)))
	tam=len(p)
	distancias=np.zeros(tam)
	pix_max_frec=[max(pontos_histog,key=operator.itemgetter(1))[0],max(pontos_histog,key=operator.itemgetter(1))[1]] 
	reta_A=pix_max_frec[1]/pix_max_frec[0]
	reta_B=-1
	reta_C=0
	i=0
	denominador=math.sqrt(reta_A**2+reta_B**2)
	while(pontos_histog[i][0]<pix_max_frec[0]):
		numerador=abs(reta_A*pontos_histog[i][0]+reta_B*pontos_histog[i][1]+reta_C)
		distancias[i]=numerador/denominador
		i=i+1
	limiar=max(list(zip(p,distancias)),key=operator.itemgetter(1))[0]
	t, img_ref = cv2.threshold(b, limiar, 255, cv2.THRESH_BINARY_INV)
	#print(limiar)
	#histograma(b,limiar)
	"""limiar = np.percentile(b, 0.22)
	histograma(b,limiar)
	print(limiar)
	#histograma(img_azul)
	t, img_ref = cv2.threshold(b, limiar, 255, cv2.THRESH_BINARY_INV)
	print(img_ref)
	#mostrar_img("Imagen en LAB",b)
	#mostrar_img("Imagen en LAB",dst)"""
	return img_ref

def agrupar(img):
	L,a,b=cv2.split(img)
	K=3
	nfil,ncol=a.shape
	img_op_a=copy.deepcopy(a)
	img_op_b=copy.deepcopy(a)
	img_op_c=copy.deepcopy(a)
	a_aux = np.array(a).flatten()
	b_aux = np.array(b).flatten()
	X=np.array(list(zip(a_aux,b_aux)))
	#X=X.T
	#cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, K, 2, error=0.005, maxiter=1000, init=None)
	#labels = np.argmax(u, axis=0)
	kmeans=KMeans(n_clusters=K).fit(X)
	labels=kmeans.predict(X)
	#print(labels.shape)
	for i in range(nfil):
		for j in range(ncol):
			grupo=labels[j+ncol*i]
			if (grupo==0):
				img_op_a[i,j]=255
			else:
				img_op_a[i,j]=0
			if (grupo==1):
				img_op_b[i,j]=255
			else:
				img_op_b[i,j]=0
			if (grupo==2):
				img_op_c[i,j]=255
			else:
				img_op_c[i,j]=0
	return img_op_a,img_op_b,img_op_c

def escolher_imagen(img_ref,img_a,img_b,img_c):
	soma_a=np.linalg.norm(img_a-img_ref)
	soma_b=np.linalg.norm(img_b-img_ref)
	soma_c=np.linalg.norm(img_c-img_ref)
	""""op_a=(1/(255*255))*(img_ref*img_a)
	op_b=(1/(255*255))*(img_ref*img_b)
	op_c=(1/(255*255))*(img_ref*img_c)

	soma_a=np.sum(op_a)
	soma_b=np.sum(op_b)
	soma_c=np.sum(op_c)"""

	#print(soma_a)
	#print(soma_b)
	#print(soma_c)
	

	suma_max=min([soma_a,soma_b,soma_c])
	if(suma_max==soma_a):
		return img_a#255*op_a
	if(suma_max==soma_b):
		return img_b#255*img_b
	if(suma_max==soma_c):
		return img_c#255*img_c

def mostrar_varias_imgs(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.axis('off')
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()