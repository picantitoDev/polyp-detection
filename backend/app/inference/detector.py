
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# Raíz del proyecto: polyp-detection/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "detector.onnx"


@dataclass(frozen=True)
class Detection:
    """Detección expresada en coordenadas del fotograma original."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


class PolypDetector:
    """Detector YOLO26s ejecutado mediante ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.25,
        providers: list[str] | None = None,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo ONNX: {self.model_path}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold debe estar entre 0 y 1."
            )

        self.confidence_threshold = confidence_threshold

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "Se esperaba un modelo con una entrada y una salida."
            )

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        # Este detector fue exportado con imgsz=640 y batch=1.
        if inputs[0].shape != [1, 3, 640, 640]:
            raise ValueError(
                f"Dimensiones de entrada inesperadas: {inputs[0].shape}"
            )

        self.input_size = 640

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, int, int]:
        """
        Convierte BGR HWC a RGB NCHW float32, con letterbox.

        Retorna:
            tensor: [1, 3, 640, 640]
            ratio: factor de escala
            pad_left: relleno horizontal
            pad_top: relleno vertical
        """
        height, width = frame.shape[:2]

        ratio = min(
            self.input_size / height,
            self.input_size / width,
        )

        new_width = round(width * ratio)
        new_height = round(height * ratio)

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        dw = self.input_size - new_width
        dh = self.input_size - new_height

        pad_left = round(dw / 2 - 0.1)
        pad_right = round(dw / 2 + 0.1)
        pad_top = round(dh / 2 - 0.1)
        pad_bottom = round(dh / 2 + 0.1)

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1),
            dtype=np.float32,
        )

        tensor /= 255.0
        tensor = tensor[np.newaxis, ...]

        return tensor, ratio, pad_left, pad_top

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Detecta pólipos en un fotograma OpenCV BGR.

        Retorna detecciones en coordenadas de la imagen original.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame debe ser un numpy.ndarray.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "El fotograma debe tener formato HWC con 3 canales BGR."
            )

        if frame.dtype != np.uint8:
            raise ValueError("El fotograma debe ser uint8.")

        height, width = frame.shape[:2]

        if height == 0 or width == 0:
            raise ValueError("El fotograma está vacío.")

        tensor, ratio, pad_left, pad_top = self._preprocess(frame)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )

        predictions = outputs[0]

        # YOLO26 end-to-end:
        # [x1, y1, x2, y2, confidence, class_id]
        # Las coordenadas se expresan en la imagen 640x640.
        if predictions.ndim != 3 or predictions.shape[0] != 1:
            raise ValueError(
                f"Forma de salida inesperada: {predictions.shape}"
            )

        if predictions.shape[2] != 6:
            raise ValueError(
                "Se esperaban 6 valores por detección."
            )

        detections: list[Detection] = []

        for prediction in predictions[0]:
            x1, y1, x2, y2, score, class_id = prediction

            if not np.isfinite(prediction).all():
                continue

            if score < self.confidence_threshold:
                continue

            # Deshacer el letterbox.
            x1 = (float(x1) - pad_left) / ratio
            y1 = (float(y1) - pad_top) / ratio
            x2 = (float(x2) - pad_left) / ratio
            y2 = (float(y2) - pad_top) / ratio

            # Limitar coordenadas al fotograma.
            x1 = max(0, min(round(x1), width))
            y1 = max(0, min(round(y1), height))
            x2 = max(0, min(round(x2), width))
            y2 = max(0, min(round(y2), height))

            # Descartar cajas sin área.
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(score),
                    class_id=int(class_id),
                )
            )

        return detections
