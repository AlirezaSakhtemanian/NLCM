import cv2 as cv
import numpy as np


def IVAR(matrix, window_size):
    k = 8
    flatten_matrix = np.ravel(matrix)
    sorted_matrix = np.sort(flatten_matrix, axis=None)[::-1]
    G_i = []
    for i in range(0, k):
        G_i.append(sorted_matrix[i])
    m_i = np.mean(flatten_matrix)
    ivar = []
    for i in range(0, k):
        ivar.append(np.power(G_i[i] - m_i, 2))
    return np.round(np.sum(ivar))


def IMEAN(matrix, window_size):
    k = 8
    flatten_matrix = np.ravel(matrix)
    sorted_matrix = np.sort(flatten_matrix, axis=None)[::-1]
    G_i = []
    for i in range(0, k):
        G_i.append(sorted_matrix[i])
    imean = np.sum(G_i) / k
    return imean


def NLCM(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    [N, M] = np.shape(gray)

    kernel_size = 5
    preprocessed = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    Window_Size = 4
    m = int(np.floor((2 * (M - Window_Size) / Window_Size) + 1))
    n = int(np.floor((2 * (N - Window_Size) / Window_Size) + 1))

    NLCM = np.zeros((n, m), dtype=np.float32)
    IMEANi = np.zeros((n, m), dtype=np.float32)
    IVARi = np.zeros((n, m), dtype=np.float32)

    row = 0
    column = 0
    for i in range(0, M - Window_Size, int(np.floor(Window_Size / 2))):
        column = 0
        for j in range(0, N - Window_Size, int(np.floor(Window_Size / 2))):
            sliding_window = preprocessed[j:j + Window_Size, i:i + Window_Size]
            IVARi[column, row] = IVAR(np.copy(sliding_window), Window_Size)
            IMEANi[column, row] = IMEAN(np.copy(sliding_window), Window_Size)
            column += 1
        row += 1

    for i in range(0, m):
        for j in range(0, n):
            IVAR_sliding_window = IVARi[j:j + 3, i:i + 3]
            if IVAR_sliding_window.shape != (3, 3):
                NLCM[j, i] = 0
                continue
            IMEAN_sliding_window = IMEANi[j:j + 3, i:i + 3]
            if IMEAN_sliding_window.shape != (3, 3):
                NLCM[j, i] = 0
                continue
            ivaru = np.ravel(IVAR_sliding_window)[4]
            IMEAN_vector = np.ravel(IMEAN_sliding_window)
            imeanu = IMEAN_vector[4]
            imeani = min(IMEAN_vector)
            NLCM[j, i] = abs((ivaru * imeanu) / imeani)

    K = 6
    std = np.floor(np.std(NLCM))
    mean = np.floor(np.mean(NLCM))
    T = mean + (K * std)

    nlcm = cv.resize(NLCM, (M, N), interpolation=cv.INTER_LINEAR)
    _, nlcm = cv.threshold(nlcm, T, 1, type=0)

    while np.sum(nlcm) >= 20 * 1:
        SE = np.array([[1, 1], [1, 1]]).astype(np.uint8)
        nlcm = cv.erode(nlcm, SE, iterations=1)

    B_size = 10
    SE_size = 3
    SE = np.ones((SE_size, SE_size), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            if nlcm[j, i] == 1:
                block = gray[j - B_size:j + B_size, i - B_size:i + B_size]
                dilated = cv.dilate(block, SE, iterations=1)
                eroded = cv.erode(block, SE, iterations=1)
                border = dilated - eroded
                std = np.mean(border)
                _, thresh = cv.threshold(border, std, 255, 0)
                x, y, w, h = cv.boundingRect(thresh.astype(np.uint8))
                cv.rectangle(image, (x + i - B_size, y + j - B_size), (x + w + i - B_size, y + h + j - B_size), (0, 0, 255), 1)
                nlcm[j - B_size:j + B_size, i - B_size:i + B_size] = 0

    return image


path =  # path to the input image
src = cv.imread(path)
output = NLCM(src)

cv.imwrite('final.png', output)
cv.imshow('final', output)
k = cv.waitKey(0)
