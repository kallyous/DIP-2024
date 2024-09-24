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

[Atividade OCR + DVC 1 - Executar pipeline de OCR](https://github.com/kallyous/DIP-OCR-MLOps-DVC)

[Atividade OCR + DVC 2- Melhorar pipeline de OCR](Ativ-Melhorar-pipeline-OCR/)

[Ativ 10 - Proc. Img. Color. 1 (TODO)](Ativ10_ProcImgColor/)

[Ativ 11 - Proc. Img. Color. 2 (Colab)](https://colab.research.google.com/drive/165M_2_ZnfId2K8nRtWOT-eVM5tx9xvp7?usp=sharing)

[Ativ 12 - Electromagnetic Spectrum (Colab)](https://colab.research.google.com/drive/1_4U9E8nTMfs_so-1gInM2TdV96XjOf_F?usp=sharing)

[Ativ 13 - Chroma Keying (Colab)](https://colab.research.google.com/drive/1Cf1i0CtZ8GjX1QqIoar6V6YR2YpDC3Ca?usp=sharing)

[Ativ 14 - HSK Disk (Colab)](https://colab.research.google.com/drive/1pcqOu0QMaJ-TBM926mD3II4Hhz7GrkE7?usp=sharing)

[Ativ 15 - Color Segmentation (Colab)](https://colab.research.google.com/drive/1Eia_efPdo5XEQl0GQw7EsSLOM_OyZS2_?usp=sharing)

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
