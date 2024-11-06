import os
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
import skimage as ski



RED = "\033[31m"
GRN = "\033[32m"
BLU = "\033[34m"
CLR = "\033[0m"

global_input_folder = "Images"
global_output_folder = "Output"
reduce_factor = 4  # Vai resultar em imagens com largura 155 e altura 202.
orig_heigth = 808
orig_width = 620



def preprocess(input_img_path: str) -> np.ndarray:
    """Preprocessamento das imagens.
    Recebe caminho para imagem e retorna uma np.ndarray contendo a imagem em HSV redimensionada."""

    # Carrega imagem. Vai vir em BGR.
    img = cv.imread(input_img_path, cv.IMREAD_COLOR)

    # Aplica suavização Gaussiana.
    img = cv.GaussianBlur(img, (5, 5), 0)

    # Binariza, pra fazermos detecção de contorno.
    img_gray = img[:, :, 2]
    threshold = ski.filters.threshold_otsu(img_gray)
    img_bin = np.where(img_gray < threshold, 0, 255).astype(np.uint8)

    # Inicializa a máscara com tudo preto.
    img_mask = np.zeros_like(img_bin)

    # Encontrar os contornos
    contours, hierarchy = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Concatenar todos os contornos encontrados em um único array de pontos.
    all_points = np.vstack(contours)

    # Casco convexo de todos os pontos de todos os contornos encontrados.
    convex_hull = cv.convexHull(all_points)
    cv.drawContours(img_mask, [convex_hull], -1, 255, thickness=cv.FILLED)  # Preenche casco com branco.

    # Calcula caixa delimitadora
    x, y, br_wi, br_he = cv.boundingRect(convex_hull)

    # Criar ROI (Region Of Interest) aplicando a máscara na imagem original.
    roi = cv.bitwise_and(img, img, mask=img_mask)

    # Calcular a posição centralizada
    img_width = img.shape[1]
    img_height = img.shape[0]
    center_x = (img_width - br_wi) // 2
    center_y = (img_height - br_he) // 2

    # Monta imagem de fundo preto com ROI da imagem original centralizado.
    img_result = np.zeros_like(img)
    img_result[center_y:center_y + br_he, center_x:center_x + br_wi] = roi[y:y + br_he, x:x + br_wi]

    # Redimensiona. Note que OpenCV usa a altura na primeira coordenada nas ndarrays.
    src_height, src_width, src_channels = img_result.shape  # src_channels é a quantidade de canais, 3.
    new_height = int(src_height / reduce_factor)
    new_width = int(src_width / reduce_factor)

    # Aplica redimensionamento da imagem. Note que nas chamadas de funções, a largura vem primeiro.
    img_result = cv.resize(img_result, (new_width, new_height))

    # Converte imagem reduzida para HSV.
    img_result = cv.cvtColor(img_result, cv.COLOR_BGR2HSV)

    # Retorna a imagem ajustada.
    return img_result



def process(input_img_path: str) -> (np.ndarray, np.ndarray):

    # Redimensiona e passa pra HSV.
    img_hsv = preprocess(input_img_path)

    # Imagem preta de mesmas dimensões que a cinza de tamanho 1/reduce_factor .
    # img_out = np.zeros((img_gray.shape[0], img_gray.shape[1]))

    # Imagem em escala de cinza, para detecção de bordas, contornos e outras operações acromáticas.
    img_gray_cable = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

    # Imagem em escala de cinza, para detecção de bordas, contornos e outras operações acromáticas.
    img_gray_stripe = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

    # DEBUG: Vamos escrever o nome da imagem original na imagem de saída, pra testar se
    #        os nomes e caminhos estão tudo certo.
    file_name = os.path.basename(input_img_path).split(".")[0]
    cable_text = f"{file_name} cable"
    stripe_text = f"{file_name} stripe"
    font = cv.FONT_HERSHEY_SIMPLEX
    font_size = 1 / 2
    font_color = (255, 255, 255)
    font_thickness = 1

    # Calcule o tamanho do texto para centralizar na imagem.
    (text_width, text_height), _ = cv.getTextSize(stripe_text, font, font_size, font_thickness)
    text_pos = ((img_hsv.shape[1] - text_width) // 2, (img_hsv.shape[0] + text_height) // 2)

    # Escreve texto na imagem do cabo.
    cv.putText(img_gray_cable, cable_text, text_pos, font, font_size, font_color, font_thickness)

    # Escreve texto na imagem da faixa.
    cv.putText(img_gray_stripe, stripe_text, text_pos, font, font_size, font_color, font_thickness)

    # Redimensiona img cabo para tamanho original.
    img_gray_cable = cv.resize( img_gray_cable, (orig_width, orig_heigth))

    # Redimensiona img cabo para tamanho original.
    img_gray_stripe = cv.resize(img_gray_stripe, (orig_width, orig_heigth))

    # Retorna a imagem processada.
    # DEBUG: Estamos restaurando o tamanho e sistema de cores originais pra visualizar o resultado
    #        do pré-processamento.
    return img_gray_cable, img_gray_stripe



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
        img_mask_cable, img_mask_stripe = process(row["src_img"])

        # Nome do arquivo de saída do cabo.
        cable_out_dir = f"{row['out_ref']}_cable.png"

        # Nome do arquivo de saída do cabo.
        stripe_out_dir = f"{row['out_ref']}_stripe.png"

        # Organiza dados numa lista de dicts.
        row_output = [ {"out_dir": cable_out_dir,  "data": img_mask_cable},
                       {"out_dir": stripe_out_dir, "data": img_mask_stripe} ]

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
        for cable, stripe in output_paths_and_values:
            cv.imwrite(cable["out_dir"], cable["data"])
            cv.imwrite(stripe["out_dir"], stripe["data"])


    print(f"{GRN}Concluido.\n{CLR}")