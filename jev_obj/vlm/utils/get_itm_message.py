import cv2
import numpy as np
from vlm.itm.blip2itm import BLIP2ITMClient

itmclient = BLIP2ITMClient(port=12182)

def get_itm_message(rgb_image, label):
    txt = f"Is there a {label} in the image?"
    cosine = itmclient.cosine(rgb_image, txt)
    itm_score = itmclient.itm_score(rgb_image, txt)
    return cosine, itm_score

def get_itm_message_cosine(rgb_image, label, room, return_metadata=False):
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    cosine, metadata = itmclient.cosine_with_metadata(rgb_image, txt)
    metadata = dict(metadata)
    metadata.update({
        "query_text": txt,
        "score_type": "blip2_itc_cosine_similarity",
        "valid": bool(metadata.get("available", False) and not metadata.get("fallback", False)),
    })
    return (cosine, metadata) if return_metadata else cosine
