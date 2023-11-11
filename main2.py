import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_statistics(data_file, sample_size, num_samples=100000):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(data_file, delimiter=';')
    
    # Initialize lists to store sample means, std devs, and variances
    sample_means = []
    sample_std_devs = []
    sample_variances = []
    
    # Perform random sampling with replacement
    for i in range(num_samples):
        sample = df.sample(n=sample_size, replace=True)
        
        if i == 0:  # Plot histogram for the first sample only
            for column in sample.columns:
                plt.figure()
                plt.hist(sample[column], bins=15, density=True, alpha=0.6)
                plt.xlabel(column)
                plt.ylabel('Density')
                plt.title(f'Histogram of {column} in the First Sample')
                plt.show()
        sample['ap_hi'] = np.clip(sample['ap_hi'], None, 300)
        sample['ap_lo'] = np.clip(sample['ap_lo'], None, 300)
        
        # Calculate mean, std dev, and variance for the sample
        sample_mean = sample.mean()
        sample_std_dev = sample.std()
        sample_variance = sample.var()
        
        # Append the results to the lists
        sample_means.append(sample_mean)
        sample_std_devs.append(sample_std_dev)
        sample_variances.append(sample_variance)
    
    # Convert lists to pandas DataFrames
    means_df = pd.DataFrame(sample_means)
    std_devs_df = pd.DataFrame(sample_std_devs)
    variances_df = pd.DataFrame(sample_variances)

    overall_mean= means_df.mean()
    overall_mean_std = means_df.std()
    
    overall_std_dev = std_devs_df.mean()
    overall_std_std =std_devs_df.std()

    overall_variance = variances_df.mean()
    overall_variance_std = variances_df.std()

    for column in means_df.columns:
        plt.figure()
        plt.hist(means_df[column], bins=25, density=True, alpha=0.6, color='b')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(f'Histogram of {column} in Sample Means (100k samples)')
        plt.show()
    
    return means_df, std_devs_df, variances_df,overall_mean,overall_std_dev,overall_variance,overall_mean_std,overall_std_std,overall_variance_std
# Load data from CSV file
data = np.genfromtxt('cardiovascular_data.csv', delimiter=';', skip_header=1)

# Extracting data columns
age = data[:, 0].astype(int)
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

ap_hi_novo = np.clip(ap_hi,0,300)
ap_lo_novo = np.clip(ap_lo,0,300)


# List of discrete variables and their names
discrete_variables = [gender, cholesterol, gluc, smoke, alco, active, cardio]
nome_discretas = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

# Display discrete variable distributions
for i, variable in enumerate(discrete_variables):
    unique_values, counts = np.unique(variable, return_counts=True)
    probabilities = counts / len(variable)
    
    plt.figure()  # Create a new figure for each plot
    plt.bar(unique_values, probabilities, label=f' Distribuicao da variavel: {nome_discretas[i]}')
    plt.xticks(unique_values)
    plt.xlabel(nome_discretas[i])
    plt.ylabel('Probabilidade')
    plt.title(f'Distribuicao da variavel: {nome_discretas[i]} ')
    plt.show()
    
    mean = np.mean(variable)
    variance = np.var(variable)
    std_dev = np.std(variable)
    print(f'Variavel:{nome_discretas[i]}\nTipo de variavel: Discreta\nGrafico: Distribuicao de probabilidade')
    print(f'E[x]: {mean}')
    print(f'V[x]: {variance}')
    print(f'DP[x]: {std_dev}')
    print('\n')

# List of continuous variables and their names
continuous_variables = [age, height, weight, ap_hi_novo, ap_lo_novo]
nome_continua = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

# Display continuous variable histograms
for i, variable in enumerate(continuous_variables):
    plt.figure()  # Create a new figure for each plot
    
    plt.hist(variable, bins=25, density=True, alpha=0.6, color='g')
    plt.xlabel(nome_continua[i])
    plt.ylabel('Densidade')
    plt.title(f'Histograma da variavel: {nome_continua[i]}')
    plt.axvline(x=np.mean(variable), color='g', linestyle='--', label=f'Média: {np.mean(variable):.2f}')
    plt.show()

    mean = np.mean(variable)
    variance = np.var(variable)
    std_dev = np.std(variable)
    print(f'Variavel:{nome_continua[i]}\nTipo de variavel: Continua\nGrafico: Histograma')
    print(f'E[x]: {mean}')
    print(f'V[x]: {variance}')
    print(f'DP[x]: {std_dev}')
    print('\n')

data_file = 'cardiovascular_data.csv'
means, std_devs, variances,means_geral,std_geral,var_geral,means_std,std_std,ver = calculate_statistics(data_file,35)
print("Media de cada amostra:")
print(means)
print("Desvio padrao de cada amostra:")
print(std_devs)
print("Variancia de cada amostra:")
print(variances)
print(f"Media das amostras:\n{means_geral}")
print(f"devio padrao das amostras:\n{std_geral}")
print(f"variancias das amostras:\n{var_geral}")
