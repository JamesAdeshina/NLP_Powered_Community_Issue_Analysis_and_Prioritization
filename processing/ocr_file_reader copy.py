import pytesseract
from PIL import Image
import docx2txt
import fitz  # PyMuPDF

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += "\n" + extract_text_from_image(img)
    return text
