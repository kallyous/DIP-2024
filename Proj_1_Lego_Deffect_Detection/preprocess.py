import math
import numpy as np
import cv2 as cv

from skimage.filters import threshold_otsu, threshold_triangle, threshold_niblack, threshold_sauvola
from skimage import img_as_ubyte

from utils import drawlines



def inverse(image_set: list):
    """Recebe imagens cinza e inverte os tons."""
    imset_out = [(255-img.copy()).astype(np.uint8) for img in image_set]
    return imset_out



def contrast_boost(image_set: list) -> list:
    """Recebe imagem cinza e aumenta contraste.
    Atualmente apenas aplica equalização de histograma.
    """
    img_out_set = []
    for img in image_set:
        img_out_set.append( cv.equalizeHist(img) )
    return img_out_set



def binarize(image_set: list, threshold_method="fixed", threshold=128, window_size=3, k=0.8) -> list:
    """Binariza a imagem."""
    img_out_set = []

    for img in image_set:

        if threshold_method == "otsu":
            t = threshold_otsu(img)

        elif threshold_method == "niblack":
            t = threshold_niblack(img, window_size=window_size, k=k)

        elif threshold_method == "sauvola":
            t = threshold_sauvola(img, window_size=window_size)

        elif threshold_method == "triangle":
            t = threshold_triangle(img)

        elif threshold_method == "fixed":
            t = threshold

        else:
            raise ValueError("Métodos de limiarização disponíveis:\n  otsu niblack sauvola triangle fixed")

        img_out_set.append(np.where(img < t, 0, 255).astype(np.uint8))

    return img_out_set



def edgesdetect_sobel(image_set: list, select="tblr"):
    """Encontra bordas em imagens binárias usando Sobel.
    O argumento 'select' indica as bordas desejadas pela primeira letra de top, bottom, left e right.
    Usar select='tl' vai retornar as bordas com branco acima e as com branco à esquerda, combinadas.
    Assim sendo, 't' implica 'quero as bordas com branco na parde de cima da borda, 'rb' implica
    'quero as bordas que branco está à direita da borda e também as bordas onde o branco está abaixo',
    e assim sucessivamente."""

    detected_borders = []
    
    for src in image_set:
        
        img = src.copy()
        img = img.astype(np.uint8)
        
        img_bord_top = np.zeros_like(img)
        img_bord_bot = np.zeros_like(img)
        img_bord_left = np.zeros_like(img)
        img_bord_right = np.zeros_like(img)

        if "t" in select:
            kernel = np.array(
                [[ 1,  2,  1],
                 [ 0,  0,  0],
                 [-1, -2, -1]]) / 4
            img_bord_top = cv.filter2D(img, -1, kernel)

        if "b" in select:
            kernel = np.array(
                [[-1, -2, -1],
                 [ 0,  0,  0],
                 [ 1,  2,  1]]) / 4
            img_bord_bot = cv.filter2D(img, -1, kernel)

        if "l" in select:
            kernel = np.array(
                [[ 1,  0, -1],
                 [ 2,  0, -2],
                 [ 1,  0, -1]]) / 4
            img_bord_left = cv.filter2D(img, -1, kernel)

        if "r" in select:
            kernel = np.array(
                [[-1,  0,  1],
                 [-2,  0,  2],
                 [-1,  0,  1]]) / 4
            img_bord_right = cv.filter2D(img, -1, kernel)

        
        # Soma as bordas com um OR binário.
        img_bord_all = img_bord_top | img_bord_bot | img_bord_left | img_bord_right

        # Adiciona à lista de bordas encontradas.
        detected_borders.append(img_bord_all)

    # Retorna lista de imagens binárias contendo as bordas detectadas.
    return detected_borders



def find_lines(image_set: list, rho=1, theta=0.0175, threshold=30):
    """Envelope para cv.HoughLines() com alguns valores default definidos."""

    lines_set = []
    
    for img in image_set:
        lines = cv.HoughLines(img, rho=rho, theta=theta, threshold=threshold)
        lines_set.append(lines)
        
    return lines_set



def search_longest_lines(image_set: list):
    """Encontra a maior linha de cada imagem."""

    # A linha mais longa possível é a diagonal principal.
    threshold = int(math.sqrt(image_set[0].shape[1]*image_set[0].shape[1] + image_set[0].shape[0]*image_set[0].shape[0]))
    
    # Solução inicial. É esperado que comece vazia.
    imset_lines = find_lines(image_set, threshold=threshold)
    
    # Faz uma busca decremental direta pra achar a melhor combinação de soluções.
    # embora seja mais interessante fazer essa busca por imagem, e faremos assim depois.
    go_next = True
    for t in range(1, threshold, 10):
        new_set = find_lines(image_set, threshold=t)
        for lines in new_set:
            try:
                line_count = len(lines)
            except TypeError:
                go_next = False
                break
            if line_count < 1: # Queremos a linha mais longa, mas é melhor duas que nenhuma.
                go_next = False
                break
        if go_next: imset_lines = new_set
        else: break

    # Retorna lista com os menores conjuntos de linhas encontrados
    return imset_lines



def setup_mask_inplace(imset_masks: list, lnset_top: list, lnset_bot: list):
    """Adiciona in-place os cortes de baixo e de cima na imagem binária fornecida."""

    # CIMA
    drawlines(imset_masks, lnset_top, inplace=True)
    for i in range(len(lnset_top)):

        line = lnset_top[i][0]
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Definir pixels acima da reta para branco
        # Obtemos a equação da reta: y = mx + b
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        b = y1 - m * x1

        # Criar uma máscara para os pixels acima da reta
        for x in range(imset_masks[i].shape[1]):
            for y in range(imset_masks[i].shape[0]):
                if y < m * x + b:  # Acima da linha
                    imset_masks[i][y, x] = 255  # Definir para branco

    # BAIXO
    drawlines(imset_masks, lnset_bot, inplace=True)
    for i in range(len(lnset_bot)):

        line = lnset_bot[i][0]
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Definir pixels acima da reta para branco
        # Obtemos a equação da reta: y = mx + b
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        b = y1 - m * x1

        # Criar uma máscara para os pixels acima da reta
        for x in range(imset_masks[i].shape[1]):
            for y in range(imset_masks[i].shape[0]):
                if y > m * x + b:  # Abaixo da linha
                    imset_masks[i][y, x] = 255  # Definir para branco

    # Retorna a referência pra própria imset_masks, mas a operação foi efetuada inplace.
    return imset_masks



def filter2D(image_set: list, kernel: np.ndarray):
    """Aplica kernel definido em uma convolução nas imagens fornecidas."""
    imset_out = []
    for img in image_set:
        imset_out.append( cv.filter2D(img, ddepth=-1, kernel=kernel) )
    return imset_out



def dilate(image_set: list, kernel_size=3, iterations=1):
    """Dilata imagens com kernel fornecido."""
    kernel = np.ones((kernel_size, kernel_size))
    imset_out = []
    for img in image_set:
        imset_out.append( cv.dilate(img.astype(np.uint8), kernel=kernel, iterations=iterations) )
    return imset_out



def erode(image_set: list, kernel_size=3, iterations=1):
    """Erode imagens com kernel fornecido."""
    kernel = np.ones((kernel_size, kernel_size))
    imset_out = []
    for img in image_set:
        imset_out.append( cv.erode(img.astype(np.uint8), kernel=kernel, iterations=iterations) )
    return imset_out



def subtract(image_set: list, mask_set: list):
    """Aplica subtração em lista de imagens."""
    return [cv.subtract(image_set[i], mask_set[i]) for i in range(len(image_set))]



def remove_table_borders(imset: list):
    """Detecta as bordas da mesa das fotos em imagem binária e a retira,
    pra facilitar as próximas operações.
    """

    # Detecção de bordas com branco no topo da borda.
    edges_top = edgesdetect_sobel(imset, "t")

    # Localiza linha mais longa nas bordas detectadas.
    top_lines = search_longest_lines(edges_top)

    # Detecção de bordas com branco abaixo da borda.
    edges_bot = edgesdetect_sobel(imset, "b")

    # Localiza linha mais longa nas bordas detectadas.
    bot_lines = search_longest_lines(edges_bot)

    # Prepara uma máscara, iniciada com zeros.
    masks = [np.zeros_like(img) for img in imset]

    # Constrói máscara para subtração.
    setup_mask_inplace(masks, top_lines, bot_lines)

    # Aplica subtração.
    masked_data = subtract(imset, masks)

    # Retorna reultado.
    return masked_data



def morph_closing(imset, kernel_size=3, iterations=1):
    """Morfologia de fechamento nas imagens fornecidas."""
    kernel = np.ones((kernel_size, kernel_size))
    out_data = [cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations) for img in imset]
    return out_data



def morph_opening(imset, kernel_size=3, iterations=1):
    """Morfologia de abertura nas imagens fornecidas."""
    kernel = np.ones((kernel_size, kernel_size))
    out_data = [cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations) for img in imset]
    return out_data



def isolate_highest_area_conn_comp(imset: list):
    """Recebe imagem binária e isola a componente conectada de maior área.
    Essa solução funciona na maioria dos casos, mas ainda temos alguns casos
    de legos que a maior comp.con. não é a do o lego desejado."""

    regions_set = []

    for img in imset:

        # Detecta componentes conexas.
        n_conn_comp, labels = cv.connectedComponents(img)

        # Vamos separar as componentes conexas.
        conn_comp = []

        # Laço sobre as componentes conexas, pelos índices. O índice 0 é o fundo, que não interessa.
        for i in range(1, n_conn_comp):

            # Prepara uma imgbin para colocar a componente.
            img_comp = np.zeros_like(img)

            # Coloca branco/255 onde a componente está presente.
            img_comp[labels == i] = 255

            # Adiciona componente na lista de componentes.
            conn_comp.append(img_comp)

        # Encontra maior componente e a adiciona às regiões.
        biggest_comp = None
        best_sum = 0
        for comp in conn_comp:
            s = comp.sum()
            if s > best_sum:
                best_sum = s
                biggest_comp = comp
        regions_set.append(biggest_comp)

    # Retorna maior região encontrada para cada imagem binária recebida.
    return regions_set



def convex_hull(imset: list):
    """Encontra e retorna cascos convexos das imagens binárias fornecidas."""

    imset_out = []

    for img in imset:

        # Encontrar os contornos
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Criar uma nova imagem para o casco convexo
        conv_hull = np.zeros_like(img)

        if contours:

            # Calcular o casco convexo para o primeiro contorno encontrado
            hull = cv.convexHull(contours[0])

            # Desenhar o casco convexo na imagem.
            cv.drawContours(conv_hull, [hull], -1, (255), thickness=cv.FILLED)

            # Adicionar à saída.
            imset_out.append(conv_hull)

    # Retornar convex hulls encontrados.
    return imset_out



def rectangle_hull(imset: list):
    """Encontra e retorna cascos convexos das imagens binárias fornecidas."""

    imset_out = []

    for img in imset:

        # Prepara máscara com zeros.
        mask = np.zeros_like(img)

        # Encontrar os contornos
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Extrai retângulo delimitador e pontos descritos.
        rotated_rect = cv.minAreaRect(contours[0])
        box = cv.boxPoints(rotated_rect)
        box = np.intp(box)

        # Desenha retângulo branco na máscara.
        cv.fillPoly(mask, [box], 255)

        # Adiciona resultado na lista de saída.
        imset_out.append(mask)

    # Retornar convex hulls encontrados.
    return imset_out



def egdes_highlight(imset: list):
    """Não tá bom. Mas a idéia é destacar as bordas. Tentar outros filtros/algoritmos para extração de bordas."""

    imset_out = []
    # Filtro Passa-Alta (High Boost Pass)
    kernel = np.array( [[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]]) / 9
    # Aplica filtro.
    edges = filter2D(imset, kernel)

    # Binariza bordas.
    edges = binarize(edges, 2)

    # Fechamento
    edges = morph_closing(edges, iterations=1)

    # Abertura
    edges = morph_opening(edges, iterations=1)

    # Adiciona à imagem original.
    for i in range(len(imset)):
        imset_out.append( cv.add(imset[i], edges[i]) )

    # Retorna imagem com bordas adicionadas.
    return imset_out



def find_contours(imset: list):
    """Recebe lista de imagens binárias {0, 255} e retorna a lista com seus contornos calculados.
    Assinatura da função envelopada:
    cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
    """
    cntrset = []
    for img in imset:
        # Equivalente a      cv2.findContours(img_bin,  cv2.RETR_LIST,      cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(img,      mode=cv.RETR_LIST,  method=cv.CHAIN_APPROX_TC89_KCOS)
        cntrset.append(contours[0])
    return cntrset



def moments(imset: list, binaryImage=True):
    """Recebe lista de contornos ou imagens binárias e retorna a lista de seus momentos.
    Por default, assume-se que imset contém imagens binárias. Se forem fornecidos contornos,
    é necessário informar com 'binaryImage=False'. Note que esse é o conportamento inverso da
    função envelopada.
    """
    momentset = []
    for img in imset:
        m = cv.moments(img, binaryImage=binaryImage)
        momentset.append(m)
    return momentset



def hu_moments(momentset: list):
    """Recebe uma lista de momentos e retorna uma lista de Hu-Momentos.
    """
    hmset = []
    for moment in momentset:
        hm = cv.HuMoments(moment)
        hmset.append(hm)
    return hmset



def extract_by_mask(maskset: list, src: list, invert_src=False, borderValue=127):
    """Recebe máscaras e imagens, e retorna os recortes feitos com as máscaras, rotacionados
    de forma a ficar o mais próximo possível da vertical dos bonecos de lego.
    """
    masked_imset = []
    img_height, img_width = maskset[0].shape
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    img_center = (img_center_x, img_center_y)

    for i in range(len(maskset)):

        # 0. Inversão de tons da imagem objeto, caso solicitada.
        if invert_src:
            img = 255 - src[i]
        else:
            img = src[i]

        # 1. Usa a máscara para extrair o objeto da imagem original.
        masked_img = cv.bitwise_and(maskset[i], img)

        # O resultado do mascaramento vem pra cá.
        img_target = np.full((img_height, img_width), borderValue, dtype=np.uint8)

        # Copiar o resultado do mascaramento pra imagem cinza.
        img_target[maskset[i] == 255] = masked_img[maskset[i] == 255]

        # 2. Extrai informações necessárias para as operações seguintes, como translação e rotação.

        # Pega os contornos da máscara, que deve ser um casco convexo do objeto detectado.
        contours, hierarchy = cv.findContours(maskset[i], mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_TC89_KCOS)

        # Pega os momentos da máscara. Pode ser feito também com contours[0] e binaryImage=False .
        moments = cv.moments(maskset[i], binaryImage=True)

        # Obtém o menor rentângulo que cabe o contorno da máscara/casco-convexo (retângulo delimitador).
        rotated_rect = cv.minAreaRect(contours[0])

        # DEBUG: Desenha o retângulo delimitador do objeto, na imagem de saída.
        # box = cv.boxPoints(rotated_rect)
        # box = np.intp(box)
        # cv.polylines(img_target, [box], isClosed=True, color=32, thickness=5)
        # print(box)

        # 3. Rotação para endireitar objeto.

        # Extraímos do retângulo delimitador as informações necessárias.
        r_center, (r_width, r_height), r_angle = rotated_rect
        # Ángulo sem rotação, pois cv2.minAreaRect() retorna o ângulo pelo qual a imagem foi rotacionada para caber
        # no menor retângulo possível descrito, mas o retângulo própriamente dito não está rotacionado.
        ur_angle = 0

        # Se o objeto está mais largo do que alto, ele está deitado, então vamos ajustar a rotação em 90º.
        if r_width > r_height:
            r_angle -= 90
            ur_angle -= 90

        # Vamos recriar o retângulo delimitador, mas com a rotação ajustada.
        # Isso será usado para recortar a imagem após ela ser rotacionada.
        unrot_rec = (r_center, (r_width, r_height), ur_angle)
        box2 = cv.boxPoints(unrot_rec)
        box2 = np.intp(box2)

        # OpenCV nos dá um atalho para calcular a matriz de rotação a partir do centro desejado e um ângulo em graus.
        # Assim não precisamos fazer na mão os cálculos em termos de radianos, seno e cosseno. Amém!
        rotation_matrix = cv.getRotationMatrix2D(r_center, r_angle, 1.0)

        # # Aplica matriz de rotação. Note que sendo mais uma matriz descrevendo função linear, usamos cv2.warpAffine() de novo.
        img_target = cv.warpAffine(img_target, rotation_matrix, (img_width, img_height), borderValue=borderValue)

        # DEBUG: Desenha o retângulo sem a rotação. Deve cobrir a imagem rotacionada.
        # cv.polylines(img_target, [box2], isClosed=True, color=16, thickness=1)

        # Coordenadas dos pontos da caixa delimitadora.
        x1, y1 = box2[0]
        x2, y2 = box2[1]
        x3, y3 = box2[2]
        x4, y4 = box2[3]

        # Queremos os maiores e menores 'x' e 'y', pra recortar a imagem.
        X = (x1, x2, x3, x4)
        Y = (y1, y2, y3, y4)
        x_start = min(X)
        x_end = max(X) + 1
        y_start = min(Y)
        y_end = max(Y) + 1

        # Recortamos a imagem.
        img_target = img_target[y_start:y_end, x_start:x_end]

        # Adiciona resultado na lista de saída.
        masked_imset.append(img_target)

    return masked_imset


    