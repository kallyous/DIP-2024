import numpy as np
import matplotlib.pyplot as plt

def create_paraboloid(size=100, scale=1.0):
    """
    Cria um paraboloide como um ndarray.

    Parâmetros:
    size (int): Define a extensão do paraboloide.
    scale (float): Escala para a altura do paraboloide.

    Retorna:
    ndarray: Uma matriz 2D representando o paraboloide.
    """
    x = np.linspace(-scale, scale, size)
    y = np.linspace(-scale, scale, size)
    x, y = np.meshgrid(x, y)
    z = x**2 + y**2
    return x, y, z

def plot_paraboloid(x, y, z):
    """
    Plota um paraboloide usando matplotlib.

    Parâmetros:
    x, y, z (ndarray): Matrizes 2D representando as coordenadas do paraboloide.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Paraboloide')
    plt.show()

# Cria e plota o paraboloide
size = 100  # Dimensão do grid
scale = 2.0  # Escala do paraboloide
x, y, z = create_paraboloid(size, scale)
plot_paraboloid(x, y, z)
