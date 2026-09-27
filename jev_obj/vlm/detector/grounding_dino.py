from pathlib import Path
from typing import Any, Optional

import numpy as np

from vlm.detector.detections import ObjectDetections

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

GROUNDING_DINO_CONFIG = str(Path(__file__).resolve().parents[2] / ".deps/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


class GroundingDINO:
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        device: Optional[Any] = None,
    ):
        import sys
        import threading
        import torch
        import torchvision.transforms.functional as transforms_functional
        from unittest.mock import patch

        dependency_path = str(
            Path(__file__).resolve().parents[2] / ".deps/GroundingDINO"
        )
        if dependency_path not in sys.path:
            sys.path.insert(0, dependency_path)
        from groundingdino.models.GroundingDINO import ms_deform_attn
        from groundingdino.util.inference import load_model, predict

        if device is None:
            device = torch.device("cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        self.model = load_model(
            model_config_path=config_path, model_checkpoint_path=weights_path
        ).to(device)
        self.caption = caption
        self.device = device
        self._torch = torch
        self._transforms_functional = transforms_functional
        self._patch = patch
        self._predict = predict
        # GroundingDINO stores transient image features on the model instance.
        # Concurrent Flask requests can replace or delete this shared state.
        self._predict_lock = threading.Lock()
        self._use_pytorch_deformable_attention = (
            self.device.type == "cuda" and not hasattr(ms_deform_attn, "_C")
        )
        if self._use_pytorch_deformable_attention:
            print(
                "GroundingDINO custom CUDA ops are unavailable; using the "
                "built-in PyTorch deformable-attention implementation on CUDA."
            )

    def predict(
        self,
        image: np.ndarray,
        caption: Optional[str] = None,
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
    ) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        image_tensor = self._transforms_functional.to_tensor(image)
        image_transformed = self._transforms_functional.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("GroundingDINO is detecting. Caption:", caption_to_use)
        with self._predict_lock, self._torch.inference_mode():
            # The dependency selects its pure-PyTorch implementation only when
            # torch.cuda.is_available() is false, even though that implementation
            # works with CUDA tensors. Scope the compatibility override tightly to
            # this serialized model forward; model and inputs remain on CUDA.
            availability = (
                self._patch.object(
                    self._torch.cuda, "is_available", return_value=False
                )
                if self._use_pytorch_deformable_attention
                else None
            )
            if availability is None:
                boxes, logits, phrases = self._predict(
                    model=self.model,
                    image=image_transformed,
                    caption=caption_to_use,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=str(self.device),
                )
            else:
                with availability:
                    boxes, logits, phrases = self._predict(
                        model=self.model,
                        image=image_transformed,
                        caption=caption_to_use,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=str(self.device),
                    )
        detections = ObjectDetections(
            boxes, logits, phrases, image_source=image, backend="grounding_dino"
        )

        # Remove detections whose class names do not exactly match the provided classes
        # classes = caption_to_use[: -len(" .")].split(" . ")
        # detections.filter_by_class(classes)

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/gdino"

    def predict(
        self,
        image_numpy: np.ndarray,
        caption: Optional[str] = "",
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12181)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
        help="inference device; defaults to CUDA when available",
    )
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(
                image,
                caption=payload["caption"],
                box_threshold=payload["box_threshold"],
                text_threshold=payload["text_threshold"],
            ).to_json()

    gdino = GroundingDINOServer(device=torch.device(args.device))
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)




