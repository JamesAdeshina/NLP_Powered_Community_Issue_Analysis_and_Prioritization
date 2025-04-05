import pytesseract
import cv2
import numpy as np
from PIL import Image
import docx2txt
import fitz  # PyMuPDF
import re


# ---- New: Preprocessing Functions ----
def preprocess_image(img):
    """Enhances image quality for OCR"""
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Grayscale conversion
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Thresholding
    _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew (simple rotation - for advanced deskew, consider angle detection)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(rotated)


# ---- New: Post-Processing ----
def clean_ocr_text(text):
    """Fixes common OCR errors"""
    text = re.sub(r'-\n', '', text)  # Join hyphenated words
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Fix extra spaces
    # Common character replacements
    replacements = {"®": "", "©": "", "™": "", "1": "I", "|": "I", "0": "O"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


# ---- Modified Functions ----
def extract_text_from_image(image):
    # Preprocess before OCR
    processed_img = preprocess_image(image)
    # Tesseract with optimized config
    custom_config = r'--oem 3 --psm 6 -l eng'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    return clean_ocr_text(text)


def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        # Extract text from rendered image (for scanned PDFs)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += extract_text_from_image(img) + "\n"
    return text