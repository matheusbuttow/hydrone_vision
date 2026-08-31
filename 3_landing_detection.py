"""
=========================================================
 3) APLICACAO DO MODELO + LOCALIZACAO 3D DA(S) BASE(S) DE POUSO
=========================================================
Roda o modelo YOLO de segmentacao treinado no script 1, acha o
centroide de cada base detectada (pode haver mais de uma) e usa a
camera ZED para converter cada centroide (pixel) em uma posicao 3D
real (X, Y, Z em metros), retornando a lista de pontos 1x por
segundo.

Requisitos:
    pip install ultralytics opencv-python numpy
    + ZED SDK instalado (https://www.stereolabs.com/developers/release)
      e o pacote python "pyzed" (vem junto com o SDK / get_python_api.py)

Dois modos de operacao (MODE):

  "zed"    (RECOMENDADO) - a propria imagem RGB da ZED e usada tanto
           para a deteccao quanto para a profundidade. Como as duas
           vem da MESMA camera, o pixel do centroide ja corresponde
           exatamente ao pixel do mapa de profundidade. Nao precisa
           de calibracao extrinseca entre camera nenhuma.

  "webcam" - a deteccao roda na webcam (calibrada no script 2) e os
           pontos sao mapeados para a resolucao da ZED por uma
           escala simples (proporcao de resolucao). Isso e uma
           APROXIMACAO: so e valida se as duas cameras estiverem
           fisicamente muito proximas, apontando para o mesmo lugar
           e com FOV parecido. Para producao de verdade, o ideal e
           fazer uma calibracao extrinseca (estereo) entre a webcam
           e a ZED. Deixado aqui porque foi o fluxo pedido
           originalmente (webcam detecta -> lista de pontos -> ZED
           aproxima profundidade).
"""

import time

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

from importlib import import_module

calib = import_module("2_camera_calibration")  # carregar_calibracao, undistort_frame


# ------------------------------------------------------------------
# CONFIGURACAO GERAL
# ------------------------------------------------------------------
MODE = "zed"  # "zed" (recomendado) ou "webcam"

WEIGHTS_PATH = "runs_pouso/base_pouso_seg/weights/best.pt"
CONF_THRESHOLD = 0.5
INTERVALO_SEGUNDOS = 1.0  # frequencia de saida da lista de pontos

WEBCAM_INDEX = 1
WEBCAM_CALIB_PATH = "webcam_calibration.npz"


# ------------------------------------------------------------------
# 1. MODELO YOLO
# ------------------------------------------------------------------
def carregar_modelo(weights_path: str) -> YOLO:
    model = YOLO(weights_path)
    return model


def obter_centroides(results, class_names, classes_alvo=None):
    """
    Extrai, para cada mascara detectada, o centroide (cx, cy) em pixels
    usando momentos de imagem (mais preciso que o centro do bounding box).

    Args:
        results: resultado unico do model.predict() (results[0])
        class_names: dict {id: nome} do modelo (model.names)
        classes_alvo: lista opcional de nomes de classe a considerar
            (ex: ["base_pouso"]). Se None, considera todas.

    Returns:
        Lista de dicts: {class_name, confidence, cx, cy}
    """
    deteccoes = []

    if results.masks is None:
        return deteccoes

    masks = results.masks.data.cpu().numpy()  # (N, H, W) em resolucao do modelo
    boxes = results.boxes
    h_img, w_img = results.orig_shape

    for i in range(masks.shape[0]):
        class_id = int(boxes.cls[i].item())
        class_name = class_names[class_id]
        confidence = float(boxes.conf[i].item())

        if classes_alvo is not None and class_name not in classes_alvo:
            continue

        mask = masks[i]
        # a mascara pode vir em resolucao diferente da imagem original -> redimensiona
        if mask.shape != (h_img, w_img):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        M = cv2.moments(mask_uint8, binaryImage=True)
        if M["m00"] == 0:
            continue  # mascara vazia, ignora

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        deteccoes.append({
            "class_name": class_name,
            "confidence": confidence,
            "cx": cx,
            "cy": cy,
        })

    return deteccoes


# ------------------------------------------------------------------
# 2. CAMERA ZED
# ------------------------------------------------------------------
def iniciar_zed(resolution=sl.RESOLUTION.HD720, fps=30):
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # ou PERFORMANCE / QUALITY
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Falha ao abrir a ZED: {status}")

    return zed


def obter_frame_zed(zed, image_mat):
    """Retorna a imagem esquerda da ZED como array BGR (numpy)."""
    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    frame_bgra = image_mat.get_data()
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    return frame_bgr


def consultar_posicao_3d(point_cloud: sl.Mat, x: int, y: int):
    """
    Consulta a nuvem de pontos da ZED no pixel (x, y) e retorna (X, Y, Z) em metros.
    Retorna (None, None, None) se o valor for invalido (sem profundidade naquele ponto).
    """
    err, point3d = point_cloud.get_value(x, y)
    if err != sl.ERROR_CODE.SUCCESS:
        return None, None, None

    X, Y, Z = float(point3d[0]), float(point3d[1]), float(point3d[2])
    if not np.isfinite([X, Y, Z]).all():
        return None, None, None

    return X, Y, Z


# ------------------------------------------------------------------
# 3. MODO ALTERNATIVO: mapear ponto da webcam para a resolucao da ZED
# ------------------------------------------------------------------
def mapear_ponto_webcam_para_zed(cx, cy, webcam_res, zed_res):
    """
    Mapeamento INGENUO por proporcao de resolucao. Assume que as duas
    cameras enxergam aproximadamente a mesma cena (mesma posicao/FOV).
    Nao substitui uma calibracao extrinseca real - use com cautela.
    """
    wx, wy = webcam_res
    zx, zy = zed_res
    scale_x = zx / wx
    scale_y = zy / wy
    return int(cx * scale_x), int(cy * scale_y)


# ------------------------------------------------------------------
# 4. UTILIDADE: escolher a "melhor" base entre varias detectadas
# ------------------------------------------------------------------
def escolher_melhor_base(pontos_pouso, frame_shape):
    """
    Entre as bases detectadas, escolhe a mais proxima do centro da
    imagem (heuristica simples: geralmente e a mais provavel de o
    drone estar sobrevoando). Ajuste essa logica conforme a missao
    (ex: priorizar classe "base_pouso" sobre "base_inicio").
    """
    validas = [p for p in pontos_pouso if p["posicao_3d_m"][2] is not None]
    if not validas:
        return None

    h, w = frame_shape[:2]
    cx_img, cy_img = w / 2, h / 2

    def dist(p):
        px, py = p["pixel"]
        return (px - cx_img) ** 2 + (py - cy_img) ** 2

    return min(validas, key=dist)


# ------------------------------------------------------------------
# 5. LOOP PRINCIPAL
# ------------------------------------------------------------------
def main():
    model = carregar_modelo(WEIGHTS_PATH)
    class_names = model.names

    zed = iniciar_zed()
    runtime_params = sl.RuntimeParameters()
    image_mat = sl.Mat()
    point_cloud = sl.Mat()

    webcam_cap = None
    webcam_camera_matrix = webcam_dist_coeffs = None
    if MODE == "webcam":
        webcam_cap = cv2.VideoCapture(WEBCAM_INDEX)
        webcam_camera_matrix, webcam_dist_coeffs = calib.carregar_calibracao(WEBCAM_CALIB_PATH)

    last_time = 0.0

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            now = time.time()
            if now - last_time < INTERVALO_SEGUNDOS:
                continue
            last_time = now

            # --- imagem usada para deteccao ---
            zed_frame = obter_frame_zed(zed, image_mat)

            if MODE == "zed":
                frame_deteccao = zed_frame
            else:
                ret, webcam_frame = webcam_cap.read()
                if not ret:
                    continue
                frame_deteccao, _ = calib.undistort_frame(
                    webcam_frame, webcam_camera_matrix, webcam_dist_coeffs
                )

            # --- inferencia (deteccao + segmentacao) ---
            results = model.predict(frame_deteccao, conf=CONF_THRESHOLD, verbose=False)[0]
            deteccoes = obter_centroides(results, class_names, classes_alvo=["base_pouso"])

            # --- profundidade / posicao 3D via ZED ---
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            pontos_pouso = []
            for det in deteccoes:
                if MODE == "zed":
                    px, py = det["cx"], det["cy"]
                else:
                    px, py = mapear_ponto_webcam_para_zed(
                        det["cx"], det["cy"],
                        webcam_res=(frame_deteccao.shape[1], frame_deteccao.shape[0]),
                        zed_res=(zed_frame.shape[1], zed_frame.shape[0]),
                    )

                X, Y, Z = consultar_posicao_3d(point_cloud, px, py)

                pontos_pouso.append({
                    "classe": det["class_name"],
                    "confianca": det["confidence"],
                    "pixel": (px, py),
                    "posicao_3d_m": (X, Y, Z),
                })

            # --- saida: lista de pontos, 1x por segundo ---
            print(f"[{time.strftime('%H:%M:%S')}] {len(pontos_pouso)} base(s) detectada(s):")
            for p in pontos_pouso:
                print(f"    {p}")

            melhor = escolher_melhor_base(pontos_pouso, zed_frame.shape)
            if melhor:
                print(f"    -> base escolhida p/ pouso: {melhor}")
                # aqui e onde voce integraria com o controlador de voo, ex:
                # enviar_setpoint_pouso(melhor["posicao_3d_m"])

            # --- visualizacao opcional ---
            vis = zed_frame.copy()
            for p in pontos_pouso:
                px, py = p["pixel"]
                cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)
                cv2.putText(vis, p["classe"], (px + 10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Deteccao Base de Pouso", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        zed.close()
        if webcam_cap is not None:
            webcam_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
