import os
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
import skimage as ski
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb



# Configurar o estilo de fundo escuro
plt.style.use('dark_background')

RED = "\033[31m"
GRN = "\033[32m"
BLU = "\033[34m"
CLR = "\033[0m"

global_input_folder = "Images"
global_output_folder = "Masks"
reduce_factor = 4  # Vai resultar em imagens com largura 155 e altura 202.
orig_heigth = 808
orig_width = 620
hue_offset = 76

# Dict de mapeamento de intervalos de matiz.
huemap = [
    {
        'name': 'Red',
        'min': 0,
        'max': 15,
        'set': np.uint8(0),
    },
    {
        'name': 'Yellow',
        'min': 16,
        'max': 45,
        'set': np.uint8(30),
    },
    {
        'name': 'Green',
        'min': 46,
        'max': 75,
        'set': np.uint8(60),
    },
    {
        'name': 'Cyan',
        'min': 76,
        'max': 105,
        'set': np.uint8(90),
    },
    {
        'name': 'Blue',
        'min': 106,
        'max': 135,
        'set': np.uint8(120),
    },
    {
        'name': 'Magenta',
        'min': 136,
        'max': 165,
        'set': np.uint8(150),
    },
]

# Corte do vermelho.
redmod = 166

def quantize_pixel(value: np.uint8) -> np.uint8:
    """Recebe o valor em [0, 180) presentando matiz, e o mapeira para cor primária mais próxima."""

    for color in huemap:
        if color['min'] <= value and value <= color['max']:
            return color['set']

    raise ValueError(f'Valor invário para matiz no OpenCV {value}')

# Cria versão vetorizada da função, pra aplicar em imagem inteira.
quantize = np.vectorize(quantize_pixel)


def show(data: dict):
    for d in data:
        cv.imshow(d['name'], d['img'])
    cv.waitKey()
    cv.destroyAllWindows()
    exit(0)



def plot_hist(data: np.ndarray, figsize=(10, 5), title=""):
    """
    Plota ndarray como histograma.

    Args:
        data:  Histograma a plotar.
        figsize (tupla widith, height):  Tamanho da figura.
        title (str):  Título para o plot.
    """

    # Criar as cores em HSV e converter para RGB
    colors_rgb = []
    for x in range(256):
        hsv = np.array([[[x % 180, 255, 255]]], dtype=np.uint8)  # HSV color
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)  # Convert to RGB
        colors_rgb.append(rgb[0, 0] / 255.0)  # Normalize RGB values for matplotlib

    plt.figure(figsize=figsize)
    plt.bar(range(256), data, color=colors_rgb)
    plt.title(title)
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()



def find_biggest_connected_component(img):

    # Detecta componentes conexas.
    n_conn_comp, labels = cv.connectedComponents(img)

    # Lista a conter as componentes conexas temporariamente pra análise.
    conn_comp = []

    # Laço sobre as componentes conexas, pelos índices. O índice 0 é o fundo, que não interessa.
    for i in range(1, n_conn_comp):

        # Prepara uma imgbin para colocar a componente.
        img_comp = np.zeros_like(img)

        # Coloca branco/255 onde a componente está presente.
        img_comp[labels == i] = 255

        # Adiciona componente na lista de componentes.
        conn_comp.append(img_comp)

    # Encontra maior componente.
    biggest_comp = None
    best_sum = 0
    for comp in conn_comp:
        s = comp.sum()
        if s > best_sum:
            best_sum = s
            biggest_comp = comp

    # Retorna maior componente encontrada.
    return biggest_comp



def preprocess(input_img_path: str) -> np.ndarray:
    """Segmentação da imagem indicada por input_img_path em máscaras binárias identificadano a ROI do objeto.
    Usa segmentação por tons de conza e por matiz.
    Recebe o caminho para a imagem. Retorna lista contendo as imagens binárias das ROI em uma lista."""

    # Carrega imagem. Vai vir em BGR.
    img = cv.imread(input_img_path, cv.IMREAD_COLOR)

    # Aplica suavização Gaussiana.
    img = cv.GaussianBlur(img, (5, 5), 0)

    # Matiz Saturação Luminosidade
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Isola camadas para preprocessar separadamente.
    layer_hue = img_hsv[:, :, 0]
    layer_sat = img_hsv[:, :, 1]
    layer_val = img_hsv[:, :, 2]

    # Máscara binária com tons.
    t = ski.filters.threshold_otsu(layer_val)
    bin_val = np.where(layer_val < t, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3))
    bin_val = cv.morphologyEx(bin_val, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
    bin_val = cv.dilate(bin_val, kernel=kernel, iterations=16)

    # DEBUG: fazendo a plotagem com matiz do histograma.
    # hue_hist = cv.calcHist([layer_hue], channels=[0], mask=bin_val, histSize=[256], ranges=[0, 256]).flatten()
    # plot_hist(hue_hist)

    # Quantiza matiz para cores primárias mais próximas de cada pixel.
    # hue_qnt = quantize(layer_hue)
    # hue_qnt_hist = cv.calcHist([hue_qnt], channels=[0], mask=bin_val, histSize=[256], ranges=[0, 256]).flatten()
    # plot_hist(hue_qnt_hist)


    # Matriz para segmentação do matiz.
    # hue_mod = layer_hue + hue_offset

    # Histograma de hue_mod
    # hue_mod_hist_plain = cv.calcHist(images=[hue_mod], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    # hue_mod_hist_plain = hue_mod_hist_plain.flatten()
    # hue_mod_hist_alter = cv.calcHist(images=[hue_mod], channels=[0], mask=bin_val, histSize=[256], ranges=[0, 256])
    # hue_mod_hist_alter = hue_mod_hist_alter.flatten()

    # Plota histogramas.
    # plot_hist(hue_mod_hist_plain, title="Hue Mod Hist Plain")
    # plot_hist(hue_mod_hist_alter, title="Hue Mod Hist Alter")

    # Intervalos de quantização. Inclusivo à esquerda, exclusivo à direita.
    # Atenção a isso, pois vai ter 256 no último elemento de Q.
    # Q = np.array([0, 31, 61, 91, 121, 151, 180]) + hue_offset
    # Usar histograma para decidir os quantis.

    # Hq = []
    # for i in range(1, len(Q)):
    #
    #     low = Q[i-1]
    #     high = Q[i]
    #
    #     # Faz bin img com faixa de matiz atual, com 0 onde o matiz não estiver presente, pra servir de fundo.
    #     #Hqi = np.where((Q[i-1] <= hue_mod and hue_mod < Q[i]), 255, 0).astype(np.uint8)
    #     Hqi = np.where((low <= hue_mod) & (hue_mod < high), 255, 0 ).astype(np.uint8)
    #
    #     # Aplica máscara de luminosidade, que é pouco mais espessa que o fio, pra caber o fio e um pouco mais.
    #     # Isso é pra não perder a listra de cor quando fica na borda e é tão escura que se confunde com o fundo.
    #     Hqi = cv.bitwise_and(Hqi, Hqi, mask=bin_val)
    #     Hqi = cv.morphologyEx(Hqi, op=cv.MORPH_CLOSE, kernel=np.ones((3, 3)), iterations=1)
    #
    #     Hq.append((f"{i}", Hqi))
    #
    #
    # Hq.append(("Hue Mod", hue_mod))
    #
    # show(Hq)

    Hq = []  # Máscaras binárias cobrindo a região de cada cor primária.
    kernel = np.ones((3, 3))
    for color in huemap:

        # Extrai imagem binária contendo o matiz atual.
        Hqi = np.where((color['min'] <= (layer_hue % redmod)) & ((layer_hue % redmod) <= color['max']), 255, 0 ).astype(np.uint8)

        # Corta com máscara de tons.
        Hqi = cv.bitwise_and(Hqi, Hqi, mask=bin_val)

        # Morfologia básica para refinar resultados e limpar ruídos.
        Hqi = cv.morphologyEx(Hqi, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
        Hqi = cv.morphologyEx(Hqi, op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)

        # Encontra componentes conexas.
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(Hqi, connectivity=4)

        # TODO: Ao invés de ficar com a maior componente, ficar com as duas maiores caso a diferença de tamanho entre
        #       elas seja menor que 50% do tamanho da maior compoenente.
        # Isso servirá para quando o fio é dividido em duas regiões pela listra.

        # Ordena lista de componentes conexas em decrescente pela área.
        enumerated_stats = enumerate(stats[1:]) # Descartamos o índice 0, pois descreve o fundo.
        enumerated_stats = list(enumerated_stats)
        enumerated_stats = sorted(enumerated_stats, key=lambda stat: stat[1][4], reverse=True)

        # Pega a componente de maior área.
        try:
            # A maior componente estará no começo da lista ordenada. Contudo,o índice está com offset para esquerda
            # por descartarmos o fundo, daí o +1.
            largest_label = enumerated_stats[0][0] + 1

        # Nessa versão, a lista estará vazia se não houver ao menos uma componente conxea para a cor atual.
        except IndexError:
            continue

        # # Encontrar maior componente conexa na img bin.
        # try:
        #     # Pulo do Gato: stats contém as estatísticas das componentes conexas. A coluna 4 de stats (stats[:, 4])
        #     # contém a área de cada componente conexa. Assim podemos encontrar o índice da maior área (ignorando
        #     # o índice 0, que é o fundo) com:
        #     largest_label = 1 + np.argmax(stats[1:, 4])  # O +1 é pq np.argmax() vai retornar o índice no contexto de
        #                                                  # [1:, 4], então o índice retornado vai estar deslocado
        #                                                  # em 1 para esquerda.
        # # Se não tiver componente conexa (i.e. tiver tudo preto), np.argmax() levanta essa exceção.
        # except ValueError:
        #     continue

        # Criar uma imagem de saída com a mesma forma, mas apenas a maior componente conexa
        Hqi = np.zeros_like(Hqi, dtype=np.uint8)

        # Em Hqi, vai definir para 255 o pixel tal que em label esse pixel é igual ao label da maior componente conexa.
        Hqi[labels == largest_label] = 255

        # Organiza dados sobre matiz atual.
        data = {
            'name': color['name'],
            'img': Hqi,
            'size': stats[largest_label][4]
        }

        # Adiciona à lista de maior componente conexa de cada matiz.
        Hq.append(data)

    # Ordena pelo tamamnho, e fica com as duas maiores
    Hq = sorted(Hq, key=lambda x: x['size'], reverse=True)[:2]

    # Morfologia básica para refinar resultado do cabo (maior região).
    m = Hq[0]['img']
    n = Hq[0]['name']
    # cv.imshow(f"{n} Raw", m)
    m = cv.morphologyEx(m, op=cv.MORPH_OPEN, kernel=kernel, iterations=5)
    m = cv.morphologyEx(m, op=cv.MORPH_CLOSE, kernel=kernel, iterations=20)
    m = cv.dilate(m, kernel=kernel, iterations=2)
    Hq[0]['img'] = m

    # Morfologia básica para refinar resultado do cabo (segunda maior região).
    m = Hq[1]['img']
    n = Hq[1]['name']
    # cv.imshow(f"{n} Raw", m)
    m = cv.morphologyEx(m, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
    m = cv.morphologyEx(m, op=cv.MORPH_CLOSE, kernel=kernel, iterations=5)
    m = cv.dilate(m, kernel=kernel, iterations=1)
    Hq[1]['img'] = m

    # Adiciona camada de tons pra ver a imagem.
    # data = {'name': "Grayscale Img",
    #         'img': layer_val,
    #         'size': -1 }
    # Hq.append(data)

    # show(Hq)

    # Retorna a imagem ajustada.
    return img_hsv, Hq



def process(input_img_path: str) -> (np.ndarray, np.ndarray):

    # Redimensiona e passa pra HSV.
    img_hsv, regions = preprocess(input_img_path)

    # Camada de tons de cinza ou luminosidade, para detecção de bordas, contornos e outras operações acromáticas.
    # layer_gray = img_hsv[:, :, 2]
    #
    # # Kernel para as operações morfológicas adiante.
    # kernel = np.ones((5, 5))
    #
    # t = 32
    # cable_init = np.where(layer_gray < t, 0, 255).astype(np.uint8)
    # cable = cv.morphologyEx(cable_init, cv.MORPH_OPEN, kernel, 1)
    #
    # t = 255 - 32
    # stripe_init = np.where(layer_gray < t, 0, 255).astype(np.uint8)
    # stripe_xor = cv.bitwise_xor(cable, stripe_init)
    # stripe = cv.morphologyEx(stripe_xor, cv.MORPH_OPEN, kernel, 1)
    #
    # # Redimensiona img cabo para tamanho original.
    # cable = cv.resize( cable, (orig_width, orig_heigth))
    #
    # # Redimensiona img cabo para tamanho original.
    # stripe = cv.resize(stripe, (orig_width, orig_heigth))
    #
    # # Retorna a imagem processada.
    # # DEBUG: Estamos restaurando o tamanho e sistema de cores originais pra visualizar o resultado
    # #        do pré-processamento.
    # return cable, stripe

    return img_hsv, regions[0]['img'], regions[1]['img']



def load_data(data_file_path: str):
    """Carrega CSV em dataframe ou encerra se arquivo não existir."""
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError as e:
        print(f"{RED}ERRO: {data_file_path} não encontrado!{CLR}\n")
        exit(2)
    return df



def find_imgs_by_id(raw_id: str) -> pd.DataFrame:
    """Recebe o id de uma amostra, tipo ID001 ou ID041, e retorna um dataframe contendo os dados de todas as imagens
    dessa amostra, como caminho do arquivo pra carregá-lo, caminho pra salvar sua saída, etc.

    Colunas do Dataframe retornado:
      sid:      id da amostra (sample id), tipo ID001 ou ID023
      folder:   nome da pasta da imagem, tipo ID001_1 ou ID001_2.
      src_img:  caminho para o arquivo fonte da imagem original.
      out_ref:  caminho definido para a saída pertinente à essa imagem, adicione sufixo em cima dessa string e salve.
    """

    # De onde carregar entrada?
    cwd = os.getcwd()
    id = raw_id.lower().strip().lstrip("id")
    id = f"ID{id}"
    id_folders = (f"{id}_1", f"{id}_2")

    # Prepara colunas do dataframe.
    cols = ["sid", "folder", "src_img", "out_ref"]
    df_out = pd.DataFrame(columns=cols)

    # Laço principal. Encontra cada imagen, carrega, processa e pareia com caminho de sua saída.
    for idf in id_folders:

        # Pastas de entrada e saída pra instância atual.
        input_folder = os.path.join(cwd, global_input_folder, idf)
        output_folder = os.path.join(cwd, global_output_folder, idf)

        # Para caso a pasta de saída ainda não exista.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Se a pasta por algum motivo não existir, apenas pule essa pasta.
        if not os.path.exists(input_folder):
            continue

        # lisca com os nomes dos arquivos dessa instância.
        filenames = [file_name for file_name in os.listdir(input_folder)
                     if os.path.isfile(os.path.join(input_folder, file_name))]

        # Opera cada imagem.
        for img_name in filenames:

            # Nome do arquivo de entrada e do arquivo de saída.
            input_img_path = os.path.join(input_folder, img_name)
            output_img_ref = os.path.join(output_folder, img_name.split(".")[0])

            # Adiciona resultado ao dataframe em construção.
            df_out.loc[len(df_out)] = [id, idf, input_img_path, output_img_ref]

    # Retorna dataframe contendo os caminhos para todas as imagens
    return df_out



def main_DL(registration_number, input_filename):
    """
    Main function to calculate the result based on input parameters.

    Args:
        registration_number (str): Student's registration number.
        input_filename (str): Absolute input file name as; <PATH + IMAGEFILE + EXTENSION> eg. 'C://User//Student//image.png' OR 'C://User//Student//image.jpg'.
                              Absolute input file name as; <PATH +  DATASET  + EXTENSION> eg. 'C://User//Student//dataset.csv' OR 'C://User//Student//dataset.npy'
    """

    print("\nDEEP LEARNING - Proj. 2\n")

    # Base de treino
    df_train = load_data("train.csv")

    # Laço nas amostras da base do CSV.
    for sample_row_index, sample_row in df_train.iterrows():

        # Pega o ID da amostra.
        id = sample_row["ID"]

        # Carrega dados das imagens nas pastas dessa amostra/Id/linha, teoricamente 8,
        # sendo 4 de IDxxx_1 e mais 4 de IDxxx_2.
        df_img_data = find_imgs_by_id(id)

        # Laço nos dados de cada uma das 8 imagens da amostra atual.
        for img_row_index, img_row in df_img_data.iterrows():

            try:
                # Carrega, redimensiona e converte a imagem para HSV.
                img_hsv = preprocess(img_row["src_img"])
            except FileNotFoundError:
                print(f"{RED}ERRO: {img_row['src_img']} não existe{CLR}")
                print("ignorando...")
                continue

            # Aqui viria a concatenação ou sei lá o que das 8 imagens para cada fio/ID, e colocar esse vetor em
            # uma lsita de vetores pra usar no treino do moedelo.

            # DEBUG: Salva a imagem (não vai fazer isso de verdade na prática, é só pra ver o resultado).
            out_hsv = img_row["out_ref"] + "_hsv.png"
            cv.imwrite(out_hsv, img_hsv)

            # DEBUG: Só pra fins de curiosidade, comparar com a imagem RGB.
            img_bgr = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
            out_rgb = img_row["out_ref"] + "_rgb.png"
            cv.imwrite(out_rgb, img_bgr)

    # Aqui devemos ter no dataframe ou em uma lista, o que for mais conveniente, os vetores com os pixels das 8 imagens
    # preprocessadas para a rede treinar e seus respectivos rótulos em cor e espessura da faixa (conforme/n]ao-conforme).
    # Assim sendo, montar e compilar o modelo aqui.
    #
    # 1. Montar modelo.
    # 2. Compilar modelo.
    # 3. Treinar modelo com dados preprocessados.
    #

    # Feito isso, usar input_filename para localizar e carregar o dataframe com os dados das imagens de validação.
    # Carregar as iamgens da mesma forma que foi para o treino, preprocessando e (se for o caso) concatenando os vetores
    # das imagens ou sei lá o que vc escolher fazer pra o vetor de atriobutos. O importante é que as imagens carregadas
    # devem ser preprocessadas da mesma forma. Depois disso faz as previsões e põe num dataframe contendo as
    # colunas [ "ID", "Cor Conforme", "Listra Conforme"].
    #
    # 1. Carregar base de validação fornecida em input_filename com preprocess(), igual ao treino.
    df_validation = load_data(input_filename)  # Carregar imagens com find_imgs_by_id(id) e preprocessá-las
    #                                            com preprocess(img_row["src_img"])
    # 2. Preparar dataframe pra receber os resultados.
    df_results = pd.DataFrame(columns=["ID", "Cor Conforme", "Listra Conforme"])
    # 3. Para cada imagem de validação:
    #     3.1. Usar modelo treinado para classificar conformidade de cor
    #     3.2. Usar modelo treinado para classificar conformidade da faixa
    # 4. Adicionar resultados dessa instância/imagem no dataframe de resultados.
    df_results.loc[len(df_results)] = ["ID069", 1, 0]
    # 5. Após processar todas as instâncias de validação, imprimir acurácia e recall.

    # Retornar o dataframee de resultados nas colunas [ "ID", "Cor Conforme", "Listra Conforme"].
    # Quando isso voltar para 'if __name__ == "__main__":' vai ser salvo como um CSV para o prof avaliar os resultados.
    #
    # 1. Retornar dataframe de resultados aqui.
    return df_results
    # Cabou.



def main_DIP(registration_number, input_filename):
    """
    Main function to calculate the result based on input parameters.

    Args:
        registration_number (str): Student's registration number.
        input_filename (str): Absolute input file name as; <PATH + IMAGEFILE + EXTENSION> eg. 'C://User//Student//image.png' OR 'C://User//Student//image.jpg'.
                              Absolute input file name as; <PATH +  DATASET  + EXTENSION> eg. 'C://User//Student//dataset.csv' OR 'C://User//Student//dataset.npy'
    """

    print("\nDIGITAL IMAGE PROCESSING - Proj. 2\n")

    # Carrega caminhos de entrada e saída pra cada arquivo do ID solicitado.
    df_paths = find_imgs_by_id(input_filename)

    # Lista de pares (caminho_saída, valor_saída)
    output_paths_and_values = []

    # Opera todas as imagens encontradas.
    for index, row in df_paths.iterrows():

        # Obtém máscaras binárias do cabo e do contorno do cabo.
        img_hsv, img_mask_cable, img_mask_stripe = process(row["src_img"])

        # Pra salvar, tem que ser em BGR/RGB
        img_rgb = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

        # Camada com pixels em valor máximo.
        layer_max = np.full_like(img_hsv[:, :, 0], 255)

        # Imagem de saída pra visualizar o matiz.
        img_hue = np.dstack([img_hsv[:, :, 0], layer_max, layer_max])
        img_hue = cv.cvtColor(img_hue, cv.COLOR_HSV2BGR)

        # Nome do arquivo de saída pra onde copiar a imagem fonte.
        img_out_dir = f"{row['out_ref']}_src.png"

        # Nome do arquivo de saída do cabo.
        cable_out_dir = f"{row['out_ref']}_cable.png"

        # Nome do arquivo de saída do cabo.
        stripe_out_dir = f"{row['out_ref']}_stripe.png"

        # Nome do arquivo de saída da camada de matiz.
        hue_out_dir = f"{row['out_ref']}_hue.png"

        # Nome do arquivo de saída da camada de seturação.
        sat_out_dir = f"{row['out_ref']}_sat.png"

        # Nome do arquivo de saída da camada de tons.
        val_out_dir = f"{row['out_ref']}_val.png"

        # Organiza dados numa lista de dicts.
        row_output = [
            {"out_dir": img_out_dir, "data": img_rgb},
            {"out_dir": cable_out_dir,  "data": img_mask_cable},
            {"out_dir": stripe_out_dir, "data": img_mask_stripe},
            {"out_dir": sat_out_dir, "data": img_hsv[:,:,1]},
            {"out_dir": val_out_dir, "data": img_hsv[:,:,2]},
            {"out_dir": hue_out_dir, "data": img_hue}
        ]

        # Adiciona resultado do processamento na lista a retornar.
        output_paths_and_values.append(row_output)

    # Retorna lista de resultados.
    return output_paths_and_values



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Project 2 - Lucas Carvalho")
    
    # Add arguments
    parser.add_argument('--registration_number', type=str, required=True, help="Student's registration number")
    parser.add_argument('--input_filename', type=str, required=True, help="Absolute input file name")

    # Parse the arguments
    args = parser.parse_args()

    # Cria pasta de saída, se não existir.
    if not os.path.exists(global_output_folder):
        os.makedirs(global_output_folder)


    # Rotina de DL.
    if args.input_filename.endswith(".csv"):

        # Call the main function with parsed arguments
        df_results = main_DL(args.registration_number, args.input_filename)

        # Write the result to file
        output_file_path = os.path.join(global_output_folder, "results.csv")
        # with open(output_file_path, "w") as out_file:
        #     out_file.write(output_value)
        df_results.to_csv(output_file_path, index=False)


    # Rotina de PID.
    else:

        # Call the main function with parsed arguments
        output_paths_and_values = main_DIP(args.registration_number, args.input_filename)

        # Write the result to file
        for src, cable, stripe, hue, sat, val in output_paths_and_values:
            cv.imwrite(src["out_dir"], src["data"])
            cv.imwrite(cable["out_dir"], cable["data"])
            cv.imwrite(stripe["out_dir"], stripe["data"])
            cv.imwrite(hue["out_dir"], hue["data"])
            cv.imwrite(sat["out_dir"], sat["data"])
            cv.imwrite(val["out_dir"], val["data"])


    print(f"{GRN}Concluido {args.input_filename}\n{CLR}")