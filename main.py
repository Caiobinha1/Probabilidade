import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(imc, title):
    plt.hist(imc, bins=100, edgecolor='black')
    plt.xlabel('IMC')
    plt.xlim(0, 100)
    plt.ylabel('Frequência')
    plt.title(title)
    plt.show()

def plot_scatter(dado1, dado2, nome):
    plt.scatter(dado1, dado2, alpha=0.5, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pressao diastolica (escala log)')
    plt.ylabel('Pressao sistolica (escala log)')
    plt.title(f'Grafico de dispersao" Diastolica x sistolica {nome}')
    plt.show()

age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio, imc, imcF, imcM, imcp, imca = ([] for _ in range(17))

data = np.genfromtxt('cardiovascular_data.csv', delimiter=';', skip_header=1)

age = (data[:, 0] / 365).astype(int)
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

for i, height_value in enumerate(height):
    if height_value / mediaaltura > 1.8 or height_value / mediaaltura < 0.5:
        print(height_value, "Está fora do normal para altura.")
    if gender[i] == 1:
        imcF.append((weight[i]) / ((height_value/100) ** 2))
    else:
        imcM.append((weight[i]) / ((height_value/100) ** 2))
    imc.append((weight[i]) / ((height_value/100) ** 2))

for i, weight_value in enumerate(weight):
    if weight_value / mediapeso > 1.8 or weight_value / mediapeso < 0.5:
        print(weight_value, "Está fora do normal para peso.")
    if cardio[i] == 0:
        imca.append((weight_value) / ((height[i]/100) ** 2))
    else:
        imcp.append((weight_value) / ((height[i]/100) ** 2))

plot_histogram(imc, 'Histograma do IMC total')
plot_histogram(imcF, 'Histograma do IMC Feminino')
plot_histogram(imcM, 'Histograma do IMC Masculino')
plot_histogram(imcp, 'Histograma do IMC com doenca cardiovascular')
plot_histogram(imca, 'Histograma do IMC sem doenca cardiovascular')
plot_scatter(ap_lo, ap_hi, 'Grafico de dispersao: Diastolica x Sistolica Total')    
