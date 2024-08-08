# DIP - 2024

Atividades da disciplina de Processamento Digital de Imagens.

Lucas Carvalho Flores  
Ufal 2024


## Atividades

[Ativ 1 - Introdução (Local)](Ativ1_Intro/)

[Ativ 2 - Manipulação de Arrays (Local)](Ativ2_ManipArrays/)

[Ativ 3 - Manipulação  de Intensidade (Local)](Ativ3_ManipIntensi/)

[Ativ 4 - Transformação de Intensidade (Local)](Ativ4_TransformIntensi/)

[Ativ 5 - Equalização de Histogramas (Local)](Ativ5_HistEqualiz/)

[Ativ 6 - Gradiente, Paraboloides e Campo gradiente (Colab)](https://colab.research.google.com/drive/1Can01egAZs3b6U5mojp-0vYwXsu3m0Nx?usp=sharing)


## Requerimentos & Instalação

#### TL; DR

```shell
pip install --user pipenv    # Instala o Pipenv
pipenv install               # Instala dependências
pipenv shell                 # Ativa ambiente virtual
cd <Pasta-da-Atividade>      # Entra no diretório da atividade
python <script-desejado>.py  # Executa script desejado
```

#### Detalhes sobre os requerimentos e instalação

Atividades feitas localmente (e não no Colab) usam o [Pipenv](https://pipenv.pypa.io/en/latest/) para gerenciamento de ambiente virtual e dependências.

* Python 3
* [Pipenv](https://pipenv.pypa.io/en/latest/)

**1.** Instalar o [Pipenv](https://pipenv.pypa.io/en/latest/) usando Pip:

```shell
pip install --user pipenv    # Instala o Pipenv
```

**2.** No diretório raiz do projeto, executar:
```bash
pipenv install               # Instala dependências
pipenv shell                 # Atiba ambiente virtual
```

**3.** Agora todas as dependências estão instaladas no ambiente virtual da raiz do projeto. Entre na pasta da atividade desejada e execute o script python normalmente.

```
cd <Pasta-da-Atividade>      # Entra no diretório da atividade
python <script-desejado>.py  # Executa script desejado
```

Exemplo:

```
pip install --user pipenv

pipenv install
pipenv shell

cd Ativ5_HistEqualiz
python ativ5a1_bit_slicing.py
```
