"""
=========================================================
 2) CALIBRACAO DA CAMERA (WEBCAM EMBARCADA NO DRONE)
=========================================================
A webcam do drone nao vem calibrada. Este modulo:
    1) Captura fotos de um tabuleiro de xadrez (chessboard).
    2) Calcula a matriz intrinseca (camera_matrix) e os
       coeficientes de distorcao (dist_coeffs).
    3) Salva o resultado em .npz para reuso no script 3
       (deteccao / localizacao da base de pouso).

OBS: a camera ZED ja vem calibrada de fabrica e se auto-calibra.
Esta calibracao aqui e so para a webcam.

Imprima um tabuleiro de xadrez (ex: 9x6 cantos internos, quadrados
de 25mm) e tire ~20-30 fotos em angulos/distancias variados,
cobrindo bem as bordas da imagem (e onde a distorcao e maior).

Requisitos:
    pip install opencv-python numpy
"""

import glob
import os

import cv2
import numpy as np


# ------------------------------------------------------------------
# 1. CAPTURA DE IMAGENS DO TABULEIRO
# ------------------------------------------------------------------
def capturar_imagens_calibracao(
    output_dir: str = "calib_images",
    camera_index: int = 1,
    n_imagens: int = 25,
    frame_width: int = 640,
    frame_height: int = 480,
):
    """Abre a webcam e salva um frame quando o usuario aperta [ESPACO]."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    count = 0
    print("Pressione [ESPACO] para capturar uma imagem do tabuleiro.")
    print("Pressione [q] para sair antes de completar.")

    while count < n_imagens:
        ret, frame = cap.read()
        if not ret:
            break

        preview = frame.copy()
        cv2.putText(preview, f"Capturas: {count}/{n_imagens}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Captura de Calibracao", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            path = os.path.join(output_dir, f"calib_{count:02d}.png")
            cv2.imwrite(path, frame)
            print(f"Salvo: {path}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_dir


# ------------------------------------------------------------------
# 2. CALCULO DA CALIBRACAO
# ------------------------------------------------------------------
def calibrar_camera(
    images_dir: str = "calib_images",
    chessboard_size: tuple = (9, 6),  # cantos internos (colunas, linhas)
    square_size_mm: float = 25.0,
):
    """Calcula camera_matrix e dist_coeffs a partir das imagens do tabuleiro."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints = []  # pontos 3D no mundo real
    imgpoints = []  # pontos 2D na imagem

    images = glob.glob(os.path.join(images_dir, "*.png")) + \
        glob.glob(os.path.join(images_dir, "*.jpg"))

    if not images:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")

    img_shape = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)

    if len(objpoints) < 5:
        raise RuntimeError(
            f"Apenas {len(objpoints)} imagens validas encontradas. "
            "Capture mais fotos do tabuleiro em angulos diferentes."
        )

    print(f"{len(objpoints)} imagens validas usadas na calibracao.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # erro de reprojecao (quanto menor, melhor - idealmente < 0.5 px)
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"Erro medio de reprojecao: {mean_error:.4f} px")

    return camera_matrix, dist_coeffs, mean_error


# ------------------------------------------------------------------
# 3. SALVAR / CARREGAR CALIBRACAO
# ------------------------------------------------------------------
def salvar_calibracao(camera_matrix, dist_coeffs, path: str = "webcam_calibration.npz"):
    np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibracao salva em: {path}")


def carregar_calibracao(path: str = "webcam_calibration.npz"):
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


# ------------------------------------------------------------------
# 4. UNDISTORT (aplicar no frame antes de rodar a YOLO)
# ------------------------------------------------------------------
def undistort_frame(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted, new_camera_matrix


if __name__ == "__main__":
    # Passo 1: capturar fotos do tabuleiro
    img_dir = capturar_imagens_calibracao(camera_index=1, n_imagens=25)

    # Passo 2: calcular calibracao
    camera_matrix, dist_coeffs, erro = calibrar_camera(
        images_dir=img_dir,
        chessboard_size=(9, 6),
        square_size_mm=25.0,
    )

    print("Camera Matrix:\n", camera_matrix)
    print("Dist Coeffs:\n", dist_coeffs)

    # Passo 3: salvar para uso no script 3 (deteccao/localizacao)
    salvar_calibracao(camera_matrix, dist_coeffs, "webcam_calibration.npz")
