from typing import Any, Optional

import numpy as np
from PIL import Image
from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image


class BLIP2ITM:
    """BLIP 2 Image-Text Matching model."""

    def __init__(
        self,
        name: str = "blip2_image_text_matching",
        model_type: str = "pretrain",
        device: Optional[Any] = None,
    ) -> None:
        import torch
        from lavis.models import load_model_and_preprocess

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name=name,
                model_type=model_type,
                is_eval=True,
                device=device,
            )
        )
        self.device = device
        self._torch = torch

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with self._torch.inference_mode():
            cosine = self.model(
                {"image": img, "text_input": txt}, match_head="itc"
            ).item()
        return cosine

    def itm_scores(self, image: np.ndarray, txt: str) -> np.ndarray:
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with self._torch.inference_mode():
            itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = self._torch.nn.functional.softmax(itm_output, dim=1)

        itm_score = itm_scores[:, 1].item()
        return itm_score


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        return self.cosine_with_metadata(image, txt)[0]

    def cosine_with_metadata(self, image: np.ndarray, txt: str):
        response = send_request(self.url, image=image, txt=txt)
        fallback = bool(response.get("fallback", False))
        metadata_present = "available" in response and "backend" in response
        metadata = {
            "available": bool(response.get("available", False)) if metadata_present else False,
            "fallback": fallback,
            "backend": str(response.get("backend", "unknown_unverified")),
            "metadata_verified": metadata_present,
        }
        return float(response["response"]), metadata

    def itm_score(self, image: np.ndarray, txt: str) -> np.ndarray:
        print(f"Question of blip2 is:{txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["itm score"])


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. 'auto' selects CUDA when it is available.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")

    print(f"Loading model on {device}...")

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {
                "response": self.cosine(image, payload["txt"]),
                "itm score": self.itm_scores(image, payload["txt"]),
                "available": True,
                "fallback": False,
                "backend": "blip2_itm",
            }

    blip = BLIP2ITMServer(device=device)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
