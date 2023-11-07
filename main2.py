import numpy as np
import matplotlib.pyplot as plt

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

discrete_variables = [gender, cholesterol, gluc, smoke, alco, active, cardio]
nome_discretas = ['gender', 'cholesterol','gluc','smoke', 'alco', 'active','cardio']
i=0
for variable in discrete_variables:
    unique_values, counts = np.unique(variable, return_counts=True)
    probabilities = counts / len(variable)
    plt.bar(unique_values, probabilities, label=f'{nome_discretas[i]} Distribution')
    plt.xticks(unique_values)
    plt.xlabel(nome_discretas[i])
    plt.ylabel('Probability')
    plt.show()
    
    mean = np.mean(variable)
    variance = np.var(variable)
    std_dev = np.std(variable)
    print(f'Variavel:{nome_discretas[i]}\nTipo de variavel: Discreta\nGrafico: Distribuicao de probabilidade')
    print(f'E[x]: {mean}')
    print(f'V[x]: {variance}')
    print(f'DP[x]: {std_dev}')
    print('\n')
    i+=1

continuous_variables = [age, height, weight, ap_hi, ap_lo]
nome_continua = ['age','height','weight','ap_hi','ap_lo']
i=0
for variable in continuous_variables:
    plt.hist(variable, bins=30, density=True, alpha=0.6, color='g')
    plt.xlabel(nome_continua[i])
    plt.ylabel('Density')
    plt.title(f'{nome_continua[i]} Histogram')
    plt.show()

    mean = np.mean(variable)
    variance = np.var(variable)
    std_dev = np.std(variable)
    print(f'Variavel:{nome_continua[i]}\nTipo de variavel: Continua\nGrafico: Histograma')
    print(f'E[x]: {mean}')
    print(f'V[x]: {variance}')
    print(f'DP[x]: {std_dev}')
    print('\n')
    i+=1