import base64
import logging
import os
import time
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_URL = "https://api-inference.huggingface.co/models/numind/NuMarkdown-8B-Thinking"

logger = logging.getLogger(__name__)


def extract_text_from_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.convert("RGB").save(buffer, format="JPEG")
        b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": b64_image}

    for attempt in range(3):
        try:
            response = requests.post(HF_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 503:
                logger.warning("Model loading (503), attempt %d/3. Waiting 20s...", attempt + 1)
                time.sleep(20)
                continue

            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            if isinstance(data, dict):
                return data.get("generated_text", "")
            return ""

        except requests.Timeout:
            logger.error("Request timed out for %s (attempt %d/3)", image_path, attempt + 1)
        except Exception as exc:
            logger.error("NuMarkdown extraction failed for %s: %s", image_path, exc)
            break

    return ""


def extract_pages(image_paths: list[str]) -> list[str]:
    results = []
    total = len(image_paths)
    for i, path in enumerate(image_paths, start=1):
        print(f"Extracting page {i} of {total}...")
        results.append(extract_text_from_image(path))
        if i < total:
            time.sleep(1)
    return results
