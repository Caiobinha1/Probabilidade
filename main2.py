import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t,norm,chi2,ttest_ind, f_oneway, chi2_contingency

def calculate_statistics(data_file, sample_size, num_samples=100000, nivel_confianca = 0.95):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(data_file, delimiter=';')
    filtrado = df[(df['cardio'] ==1)]
    #filtered_data = df[(df['gender'] == 1) & (df['smoke'] == 1) & (df['age'] > 16200)] #para 3.1.5

    # Initialize lists to store sample means, std devs, and variances
    sample_means = []
    sample_std_devs = []
    sample_variances = []
    
    # Perform random sampling with replacement
    for i in range(num_samples):
        sample = filtrado.sample(n=sample_size, replace=True)  #filtered!! se quiser o todo usamos df
        

        
        if i == 0:  # Plot histogram for the first sample only
            #for column in sample.columns:
            #    plt.figure()
            #    plt.hist(sample[column], bins=15, density=True, alpha=0.6)
            #    plt.xlabel(column)
            #    plt.ylabel('Density')
            #    plt.title(f'Histogram of {column} in the First Sample')
            #    plt.show()
            sample_mean = sample.mean()
            sample_std_dev = sample.std()
            sample_variance = sample.var()
            
            n = 45
            chi2_lower = chi2.ppf(0.05/2, df=n-1)
            chi2_upper = chi2.ppf(0.95/2, df=n-1)

            # Intervalo de confiança para a variância
            confidence_interval = ((n-1) * sample_variance['weight'] / chi2_upper, (n-1) * sample_variance['weight'] / chi2_lower)

            #print(f"Intervalo de Confiança para a Variância: {confidence_interval}\n{sample_variance['weight']}")
            '''
            filtrado = sample[(sample['gender'] == 2) & (sample['age'] > 18000) & (sample['smoke'] == 1)]        #3.1.7
            proporcao = (len(filtrado))/sample_size
            print(len(filtrado), proporcao)
            z = norm.ppf(0.975) #alpha/2
            margin_of_error = z * np.sqrt((proporcao * (1 - proporcao)) / sample_size)
            confidence_interval = (proporcao - margin_of_error, proporcao + margin_of_error)

            print(f"Intervalo de Confiança para a proporção: {confidence_interval}")
            
            
            critical_value = t.ppf((1 + nivel_confianca) / 2, df=sample_size - 1)

            ap_hi_ci = [sample_mean['ap_hi'] - critical_value * sample_std_dev['ap_hi'] / np.sqrt(sample_size),
                sample_mean['ap_hi'] + critical_value * sample_std_dev['ap_hi'] / np.sqrt(sample_size)]

            ap_lo_ci = [sample_mean['ap_lo'] - critical_value * sample_std_dev['ap_lo'] / np.sqrt(sample_size),
                sample_mean['ap_lo'] + critical_value * sample_std_dev['ap_lo'] / np.sqrt(sample_size)]

            print(f"Intervalo de Confiança para ap_hi: {ap_hi_ci}")
            print(f"Intervalo de Confiança para ap_lo: {ap_lo_ci}")
            '''
        
        sample['ap_hi'] = np.clip(sample['ap_hi'], None, 300)
        sample['ap_lo'] = np.clip(sample['ap_lo'], None, 300)
                # Calcular o intervalo de confiança para as médias de ap_hi e ap_lo

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

        # Calcular o intervalo de confiança para as médias de ap_hi e ap_lo
    critical_value = t.ppf((1 + nivel_confianca) / 2, df=sample_size - 1)

    ap_hi_ci = [overall_mean['ap_hi'] - critical_value * overall_mean_std['ap_hi'] / np.sqrt(sample_size),
                overall_mean['ap_hi'] + critical_value * overall_mean_std['ap_hi'] / np.sqrt(sample_size)]

    ap_lo_ci = [overall_mean['ap_lo'] - critical_value * overall_mean_std['ap_lo'] / np.sqrt(sample_size),
                overall_mean['ap_lo'] + critical_value * overall_mean_std['ap_lo'] / np.sqrt(sample_size)]

    print(f"Intervalo de Confiança para ap_hi: {ap_hi_ci}")
    print(f"Intervalo de Confiança para ap_lo: {ap_lo_ci}")
    
    return means_df, std_devs_df, variances_df,overall_mean,overall_std_dev,overall_variance,overall_mean_std,overall_std_std,overall_variance_std

def test_proporcao_alcool(data_file, sample_size=300, alpha=0.05):
    # Leitura do CSV
    df = pd.read_csv(data_file, delimiter=';')
    filtrado = df[(df['alco'] == 1)]
    # Amostragem aleatória
    sample = filtrado.sample(n=sample_size, replace=True)
    
    # Filtragem por gênero
    homens = sample[sample['gender'] == 2]['alco']
    mulheres = sample[sample['gender'] == 1]['alco']
    
    # 4.
    p_homens = len(homens)/sample_size       
    p_mulheres = len(mulheres)/sample_size
    print(f"Proporcao homens: {p_homens}\nProporcao mulheres: {p_mulheres}\n{len(homens)} {len(mulheres)}")
    
    z_score = (p_homens - p_mulheres) / np.sqrt(((p_homens * (1 - p_homens))/len(homens)) + ((p_mulheres * (1 - p_mulheres))/len(mulheres)))
    
    p_value = 1 - norm.cdf(z_score)
    print(f"Valor p= {p_value}")

    # 5. Decisao
    reject_null = p_value < alpha
    
    # 6. Conclua o teste com base nas evidências da amostra
    if reject_null:
        print("Rejeitamos a hipótese nula.")
        print("Há evidências de que a proporção de homens que ingerem bebida alcoólica é maior que a de mulheres.")
    else:
        print("Não rejeitamos a hipótese nula.")
        print("Não há evidências suficientes para afirmar que a proporção de homens que ingerem bebida alcoólica é maior que a de mulheres.")

def test_cardio_weight(data_file, alpha=0.05):
    # Leitura do CSV
    df = pd.read_csv(data_file, delimiter=';')
    
    # Filtragem por doenças cardiovasculares
    com_cardio = df[df['cardio'] == 1]['weight']
    sem_cardio = df[df['cardio'] == 0]['weight']
    
    # Teste de Diferença de Médias (t-test)
    t_stat, p_value_mean = ttest_ind(com_cardio, sem_cardio, equal_var=False)
    
    # Teste de Diferença de Variâncias (F-test)
    f_stat, p_value_var = f_oneway(com_cardio, sem_cardio)
    
    # Teste de Diferença de Proporções (Chi-square)
    contingency_table = pd.crosstab(df['cardio'], df['weight'])
    chi2_stat, p_value_prop, _, _ = chi2_contingency(contingency_table)
    
    # Decisões com base nos valores p
    reject_mean = p_value_mean < alpha
    reject_var = p_value_var < alpha
    reject_prop = p_value_prop < alpha
    
    # Resultados
    print("Teste de Diferença de Médias:")
    print(f"T-Stat: {t_stat}")
    print(f"Valor-P: {p_value_mean}")
    print("Rejeitamos a hipótese nula." if reject_mean else "Não rejeitamos a hipótese nula.")
    print("\nTeste de Diferença de Variâncias:")
    print(f"F-Stat: {f_stat}")
    print(f"Valor-P: {p_value_var}")
    print("Rejeitamos a hipótese nula." if reject_var else "Não rejeitamos a hipótese nula.")
    print("\nTeste de Diferença de Proporções:")
    print(f"Chi2-Stat: {chi2_stat}")
    print(f"Valor-P: {p_value_prop}")
    print("Rejeitamos a hipótese nula." if reject_prop else "Não rejeitamos a hipótese nula.")



df = pd.read_csv('cardiovascular_data.csv', delimiter=';')

#filtered_data = df[(df['gender'] == 2) & (df['age'] > 18000)] #para 3.1.5
#varamostra = filtered_data.var()
# vari = varamostra['height']
#filtered_data = df[(df['gender'] == 2) & (df['age'] > 18000) & (df['smoke'] == 1)] #para 3.1.6
#proportion = len(filtered_data)/70000
#print(proportion)
filtered_data = df[(df['cardio'] ==1)]
varamostra = filtered_data.var()
vari = varamostra['weight']
#print(vari)

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
 
"""
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
"""
data_file = 'cardiovascular_data.csv'
test_proporcao_alcool(data_file)
test_cardio_weight(data_file)
means, std_devs, variances,means_geral,std_geral,var_geral,means_std,std_std,ver = calculate_statistics(data_file,45)

print("Media de cada amostra:")
print(means)
print("Desvio padrao de cada amostra:")
print(std_devs)
print("Variancia de cada amostra:")
print(variances)
print(f"Media das amostras:\n{means_geral}")
print(f"devio padrao das amostras:\n{std_geral}")
print(f"variancias das amostras:\n{var_geral}")

