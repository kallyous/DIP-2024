"""Carrega imagem em escala de cinza e plota seu histograma.
Compara as plotagens do pyplot e seaborn.
Lucas Carvalho Flores
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Parâmetros da distribuição normal
mean = 0       # Média
std_dev = 1    # Desvio padrão
num_samples = 1000  # Número de amostras

# Gerar dados de uma distribuição normal
data = np.random.normal(mean, std_dev, num_samples)


""" MATPLOTLIB """

# Criar uma figura
plt.figure()

# Plotar o histograma dos dados
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Dados')

# Plotar a função de densidade de probabilidade (PDF) da distribuição normal
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2, label='Distribuição Normal')

# Adicionar títulos e rótulos
plt.title('Distribuição Normal')
plt.xlabel('Valor')
plt.ylabel('Densidade de Probabilidade')
plt.legend()

# Salvar a figura
plt.savefig('distribuicao_normal.png')

# Exibir a figura
plt.show()


""" SEABORN """

# Criar uma figura
plt.figure()

# Plotar o histograma dos dados e a KDE (Kernel Density Estimate) com seaborn
sns.histplot(data, bins=30, kde=True, color='g', label='Dados')

# Adicionar títulos e rótulos
plt.title('Distribuição Normal')
plt.xlabel('Valor')
plt.ylabel('Densidade de Probabilidade')
plt.legend()

# Salvar a figura
plt.savefig('distribuicao_normal_seaborn.png')

# Exibir a figura
plt.show()
