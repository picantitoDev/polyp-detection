
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from app.inference.detector import Detection, PolypDetector
from app.inference.classifier import Classification, PolypClassifier


@dataclass(frozen=True)
class PolypPrediction:
    """Detección de un pólipo y su clasificación asociada."""

    detection: Detection
    classification: Classification


@dataclass(frozen=True)
class InferenceMetrics:
    """Duración de las etapas de inferencia en milisegundos."""

    detection_ms: float
    classification_ms: float
    total_ms: float


@dataclass(frozen=True)
class InferenceResult:
    """Resultado completo del procesamiento de un fotograma."""

    frame: np.ndarray
    predictions: list[PolypPrediction]
    metrics: InferenceMetrics


class PolypInferencePipeline:
    """
    Pipeline de detección y clasificación de pólipos.

    Entrada:
        Fotograma OpenCV BGR uint8.

    Salida:
        Fotograma anotado, predicciones y tiempos.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        crop_padding: float = 0.15,
        detector: PolypDetector | None = None,
        classifier: PolypClassifier | None = None,
    ):
        if crop_padding < 0:
            raise ValueError("crop_padding no puede ser negativo.")

        self.crop_padding = crop_padding

        # Se cargan una sola vez y se reutilizan.
        self.detector = detector or PolypDetector(
            confidence_threshold=confidence_threshold
        )

        self.classifier = classifier or PolypClassifier()

    def _crop_with_padding(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> np.ndarray:
        """
        Extrae un recorte alrededor de la detección
        incluyendo un margen adicional de contexto.
        """
        height, width = frame.shape[:2]

        box_width = detection.x2 - detection.x1
        box_height = detection.y2 - detection.y1

        pad_x = round(box_width * self.crop_padding)
        pad_y = round(box_height * self.crop_padding)

        x1 = max(0, detection.x1 - pad_x)
        y1 = max(0, detection.y1 - pad_y)
        x2 = min(width, detection.x2 + pad_x)
        y2 = min(height, detection.y2 + pad_y)

        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _draw_prediction(
        frame: np.ndarray,
        prediction: PolypPrediction,
    ) -> None:
        """Dibuja la caja y la clasificación sobre el fotograma."""
        detection = prediction.detection
        classification = prediction.classification

        # OpenCV utiliza BGR.
        if classification.class_id == 0:
            color = (0, 165, 255)  # Naranja: adenoma
            label = "ADENOMA"
        else:
            color = (0, 200, 0)  # Verde: hiperplásico
            label = "HIPERPLASICO"

        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )

        text = (
            f"{label} {classification.confidence:.2f}"
            f" | DET {detection.confidence:.2f}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        frame_height, frame_width = frame.shape[:2]

        # Evitar que la etiqueta salga de la imagen.
        text_x = min(
            detection.x1,
            max(0, frame_width - text_width - 8),
        )

        if detection.y1 >= text_height + baseline + 12:
            box_top = detection.y1 - text_height - baseline - 10
        else:
            box_top = min(
                detection.y2 + 4,
                max(0, frame_height - text_height - baseline - 8),
            )

        box_bottom = min(
            frame_height,
            box_top + text_height + baseline + 8,
        )

        cv2.rectangle(
            frame,
            (text_x, box_top),
            (
                min(frame_width, text_x + text_width + 8),
                box_bottom,
            ),
            color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (text_x + 4, box_top + text_height + 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_watermark(frame: np.ndarray) -> None:
        """Indica que el sistema es un prototipo experimental."""
        text = "MVP EXPERIMENTAL - NO USO CLINICO"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        frame_height, frame_width = frame.shape[:2]

        x = 8
        y = frame_height - 10

        cv2.rectangle(
            frame,
            (0, y - text_height - 7),
            (min(frame_width, text_width + 16), frame_height),
            (25, 25, 25),
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> InferenceResult:
        """
        Ejecuta detección, clasificación y anotación.

        No modifica el fotograma original.
        """
        start_total = perf_counter()

        # 1. Detección.
        start_detection = perf_counter()

        detections = self.detector.predict(frame)

        detection_ms = (
            perf_counter() - start_detection
        ) * 1000.0

        # 2. Extraer recortes.
        valid_detections: list[Detection] = []
        crops: list[np.ndarray] = []

        for detection in detections:
            crop = self._crop_with_padding(frame, detection)

            if crop.size == 0:
                continue

            valid_detections.append(detection)
            crops.append(crop)

        # 3. Clasificación por lotes.
        start_classification = perf_counter()

        classifications = self.classifier.predict(crops)

        classification_ms = (
            perf_counter() - start_classification
        ) * 1000.0

        if len(classifications) != len(valid_detections):
            raise RuntimeError(
                "El número de clasificaciones no coincide "
                "con las detecciones."
            )

        predictions = [
            PolypPrediction(
                detection=detection,
                classification=classification,
            )
            for detection, classification in zip(
                valid_detections,
                classifications,
            )
        ]

        # 4. Dibujar anotaciones sobre una copia.
        annotated_frame = frame.copy()

        for prediction in predictions:
            self._draw_prediction(
                annotated_frame,
                prediction,
            )

        self._draw_watermark(annotated_frame)

        total_ms = (
            perf_counter() - start_total
        ) * 1000.0

        return InferenceResult(
            frame=annotated_frame,
            predictions=predictions,
            metrics=InferenceMetrics(
                detection_ms=detection_ms,
                classification_ms=classification_ms,
                total_ms=total_ms,
            ),
        )
