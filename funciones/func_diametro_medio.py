import cv2
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt

def get_diametro_medio_celulas(img):
	img_filtrada = reduce_noise(img)
	img_invert = img_filtrada
	ret, img_binaria = cv2.threshold(np.uint8(img_invert), 0, 255, cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	img_fundo_limpo = cv2.morphologyEx(img_binaria, cv2.MORPH_OPEN, kernel)
	img_sem_buracos = 255 * ndimage.binary_fill_holes(img_fundo_limpo).astype(int)
	img_sem_buracos_diam = np.uint8(img_sem_buracos.copy())
	diametro = diametro_medio(img_sem_buracos_diam)

	return diametro

def diametro_medio(img):
	i=3
	d=2
	porc_pix_brancos=1
	lista_porc=[]
	lista_tam_SE=[]
	lista_dif=[]
	num_tot_pix_img=(img.shape[0])*(img.shape[1])
	while(porc_pix_brancos!=0):
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
		img_fin = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
		porc_pix_brancos=((np.sum(img_fin))/255)/num_tot_pix_img
		lista_porc.append(porc_pix_brancos)
		lista_tam_SE.append(i)
		i=i+d
	for i in range(0,len(lista_tam_SE)-1):
		dif=lista_porc[i]-lista_porc[i+1]
		lista_dif.append(dif)
	lista_tam_SE.pop(len(lista_tam_SE)-1)
	dimensao=lista_tam_SE[lista_dif.index(max(lista_dif))]
	diametro=dimensao+2

	return diametro

def reduce_noise(img):
	tam=801
	nfil,ncol=img.shape
	img=img.astype(np.float32)	
	img_fgauss=cv2.GaussianBlur(img,(tam,tam),0,0,cv2.BORDER_REFLECT)
	img_final=img-img_fgauss

	return img_final

def histograma(img):
	hist = cv2.calcHist([img], [0], None, [256], [0, 256])
	plt.plot(hist, color='gray' )
	plt.xlabel('Intensidade de iluminação')
	plt.ylabel('quantidade de pixeles')
	plt.show()

def mostrar_img(titulo,img):
	img=np.uint8(img)
	plt.imshow(img, cmap='gray' )
	plt.title(titulo)
	plt.show()

	