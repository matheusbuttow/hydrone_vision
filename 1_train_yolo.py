"""
=========================================================
 1) TREINAMENTO DO MODELO YOLOv11 (SEGMENTACAO)
=========================================================
Treina um modelo YOLOv11 (Ultralytics) para SEGMENTAR a base de pouso
do drone, ja configurando os parametros de data augmentation
(brilho, contraste, saturacao, rotacao, flips, mosaic, etc).

Requisitos:
    pip install ultralytics
"""

from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------
# 1. CONFIGURACAO DE DATA AUGMENTATION
# ------------------------------------------------------------------
AUGMENTATION_PARAMS = {
    # --- cor / iluminacao (contraste, brilho, saturacao) ---
    "hsv_h": 0.015,   # variacao de matiz (0-1)
    "hsv_s": 0.7,     # variacao de saturacao 
    "hsv_v": 0.4,     # variacao de brilho/valor 

    # --- geometria ---
    "degrees": 10.0,      # rotacao maxima (graus)
    "translate": 0.1,     # translacao (fracao da imagem)
    "scale": 0.5,         # zoom in/out
    "shear": 2.0,         # distorcao angular
    "perspective": 0.0005,# distorcao de perspectiva
    "flipud": 0.3,        # flip vertical
    "fliplr": 0.5,        # flip horizontal

    # --- augmentations compostas ---
    "mosaic": 1.0,      # combina 4 imagens em 1
    "mixup": 0.1,       # mistura duas imagens
    "copy_paste": 0.1,   # copia objetos entre imagens

    "erasing": 0.2,   # apaga regioes aleatorias (oclusao)
}


def train_model(
    data_yaml: str,
    model_arch: str = "yolo11n-seg.pt", 
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs_pouso",
    name: str = "base_pouso_seg_yolo11",
    device: str = "0",
    resume: bool = False,
):
    """Treina o modelo YOLOv11 de segmentacao com os augmentations definidos."""
    assert Path(data_yaml).exists(), f"data.yaml nao encontrado: {data_yaml}"

    model = YOLO(model_arch)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        resume=resume,
        patience=30,  # early stopping
        val=True,
        **AUGMENTATION_PARAMS,
    )
    return results, model


def validate_model(model: YOLO, data_yaml: str):
    """Roda validacao e imprime metricas (mAP, precisao, recall)."""
    metrics = model.val(data=data_yaml)
    print(metrics)
    return metrics


def export_model(model: YOLO, fmt: str = "onnx", imgsz: int = 640):
    """Exporta o modelo treinado (ex: para ONNX)."""
    path = model.export(format=fmt, imgsz=imgsz, simplify=True)
    print(f"Modelo exportado em: {path}")
    return path


if __name__ == "__main__":
    DATA_YAML = "data.yaml"

    results, model = train_model(
        data_yaml=DATA_YAML,
        model_arch="yolo11n-seg.pt",  # <--- Aqui definimos a YOLOv11 Nano de segmentação
        epochs=50,
        imgsz=640,
        batch=16,
        device="cpu",
    )

    validate_model(model, DATA_YAML)

    # O arquivo best.pt será salvo em runs_pouso/base_pouso_seg_yolo11/weights/best.pt
    export_model(model, fmt="onnx", imgsz=640)