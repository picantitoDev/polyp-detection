
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "classifier.onnx"

CLASS_NAMES = ("adenoma", "hiperplasico")

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(3, 1, 1)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(3, 1, 1)


@dataclass(frozen=True)
class Classification:
    """Resultado de clasificación de un recorte de pólipo."""

    class_id: int
    label: str
    confidence: float
    adenoma_probability: float
    hyperplastic_probability: float


class PolypClassifier:
    """Clasificador ConvNeXt-Tiny mediante ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        providers: list[str] | None = None,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo ONNX: {self.model_path}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "Se esperaba una entrada y una salida ONNX."
            )

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        input_shape = inputs[0].shape
        output_shape = outputs[0].shape

        if input_shape[1:] != [3, 224, 224]:
            raise ValueError(
                f"Entrada ONNX inesperada: {input_shape}"
            )

        if output_shape[-1] != 2:
            raise ValueError(
                f"Salida ONNX inesperada: {output_shape}"
            )

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento equivalente al de evaluación
        de torchvision ConvNeXt_Tiny_Weights.DEFAULT.

        Entrada:
            Imagen BGR uint8 de OpenCV.

        Salida:
            Tensor float32 [3, 224, 224].
        """
        if not isinstance(crop, np.ndarray):
            raise TypeError("El recorte debe ser un numpy.ndarray.")

        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(
                "El recorte debe tener formato HWC con 3 canales BGR."
            )

        if crop.dtype != np.uint8:
            raise ValueError("El recorte debe ser uint8.")

        height, width = crop.shape[:2]

        if height == 0 or width == 0:
            raise ValueError("El recorte está vacío.")

        # OpenCV BGR -> PIL RGB.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Resize: lado más corto = 236 píxeles.
        original_width, original_height = image.size

        if original_width <= original_height:
            resized_width = 236
            resized_height = int(
                236 * original_height / original_width
            )
        else:
            resized_height = 236
            resized_width = int(
                236 * original_width / original_height
            )

        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )

        # Center crop: 224x224.
        left = (resized_width - 224) // 2
        top = (resized_height - 224) // 2

        image = image.crop(
            (left, top, left + 224, top + 224)
        )

        # PIL RGB -> float32 [0, 1].
        array = np.asarray(image, dtype=np.float32) / 255.0

        # HWC -> CHW.
        tensor = array.transpose(2, 0, 1)

        # Normalización ImageNet.
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

        return np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax numéricamente estable."""
        shifted = logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )

        exp_values = np.exp(shifted)

        return exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )

    def predict(
        self,
        crops: list[np.ndarray],
    ) -> list[Classification]:
        """
        Clasifica una lista de recortes BGR.

        Retorna un resultado por recorte, en el mismo orden.
        """
        if not crops:
            return []

        tensors = [
            self._preprocess(crop)
            for crop in crops
        ]

        batch = np.stack(tensors, axis=0)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )

        logits = np.asarray(outputs[0])

        if logits.shape != (len(crops), 2):
            raise ValueError(
                f"Forma de salida inesperada: {logits.shape}"
            )

        if not np.isfinite(logits).all():
            raise ValueError(
                "El modelo devolvió valores no finitos."
            )

        probabilities = self._softmax(logits)

        results: list[Classification] = []

        for probs in probabilities:
            class_id = int(np.argmax(probs))

            results.append(
                Classification(
                    class_id=class_id,
                    label=CLASS_NAMES[class_id],
                    confidence=float(probs[class_id]),
                    adenoma_probability=float(probs[0]),
                    hyperplastic_probability=float(probs[1]),
                )
            )

        return results
