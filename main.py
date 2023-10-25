
import numpy as np
import matplotlib.pyplot as plt


def printhistogramas1(imc,imcF,imcM):
    plt.hist(imc, bins=100, edgecolor='black')  # `bins` define o número de barras no histograma
    plt.xlabel('IMC')  
    plt.xlim(0, 100)
    plt.ylabel('Frequência')  
    plt.title('Histograma do IMC total')  
    plt.show()  

    plt.hist(imcF, bins=100, edgecolor='black')  
    plt.xlim(0, 100)
    plt.ylabel('Frequência')  
    plt.title('Histograma do IMC Feminino')  
    plt.show()  

    plt.hist(imcM, bins=100, edgecolor='black')  
    plt.xlabel('IMC')  
    plt.xlim(0, 100)
    plt.ylabel('Frequência')  
    plt.title('Histograma do IMC Masculino')  
    plt.show() 

def printhistogramas2(imcp,imca):
    plt.hist(imcp, bins=100, edgecolor='black') 
    plt.xlabel('IMC')  
    plt.xlim(0, 100)
    plt.ylabel('Frequência')  
    plt.title('Histograma do IMC com presenca de doenca cardiovascular')  
    plt.show()  

    plt.hist(imca, bins=100, edgecolor='black')  
    plt.xlabel('IMC')  
    plt.xlim(0, 100)
    plt.ylabel('Frequência')  
    plt.title('Histograma do IMC com ausencia de doenca cardiovascular')  
    plt.show() 
age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio, imc, imcF, imcM, imcp, imca = ([] for _ in range(17))


data = np.genfromtxt('cardiovascular_data.csv', delimiter=';', skip_header=1)


age = (data[:, 0] / 365).astype(int)  # Transforma dias em anos
gender = data[:, 1].astype(int)
height = data[:, 2].astype(int)
weight = data[:, 3].astype(float)
ap_hi = data[:, 4].astype(int)
ap_lo = data[:, 5].astype(int)
cholesterol = data[:, 6].astype(int)
gluc = data[:, 7].astype(int)
smoke = data[:, 8].astype(int)
alco = data[:, 9].astype(int)
active = data[:, 10].astype(int)
cardio = data[:, 11].astype(int)


mediaaltura = np.mean(height)
mediapeso = np.mean(weight)

# Verifica e imprime os valores fora do normal para altura e peso
for i, height_value in enumerate(height):
    if height_value / mediaaltura > 2 or height_value / mediaaltura < 0.2:
        print(height_value, "Está fora do normal para altura.") # altura fora do normal
    if gender[i]==1:
        imcF.append((weight[i]) / ((height_value/100) ** 2))
    else:
        imcM.append((weight[i]) / ((height_value/100) ** 2))
    imc.append((weight[i]) / ((height_value/100) ** 2))            #imc de cada pessoa



for i, weight_value in enumerate(weight):
    if weight_value / mediapeso > 2 or weight_value / mediapeso < 0.2:
        print(weight_value, "Está fora do normal para peso.")   # peso fora do normal
    if cardio[i]==0:
        imca.append((weight_value) / ((height[i]/100) ** 2))
    else:
        imcp.append((weight_value) / ((height[i]/100) ** 2))
    

printhistogramas1(imc,imcF,imcM)
printhistogramas2(imcp,imca)

