import csv
import os
import unicodedata
from fpdf import FPDF
from docx import Document

def normalize_text(text):
    """
    Normalize the text to ASCII by removing characters that cannot be encoded in Latin-1.
    Note: This will remove/replace some Unicode characters.
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# Define output folders for PDF and DOC files
pdf_folder = "pdf_letters"
doc_folder = "doc_letters"

# Create directories if they don't exist
os.makedirs(pdf_folder, exist_ok=True)
os.makedirs(doc_folder, exist_ok=True)

# Path to your CSV file
csv_file = "letters.csv"

with open(csv_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for idx, row in enumerate(reader, start=1):
        letter_text = row["text"]
        # Normalize the text to remove characters not supported by Latin-1
        normalized_text = normalize_text(letter_text)
        sentiment = row["sentiment"]
        category = row["category"]

        # Create a base filename (you can adjust naming convention as needed)
        base_filename = f"letter_{idx}_{sentiment}_{category}".replace(" ", "_")

        # ---- Generate PDF File using default font ----
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, normalized_text)
        pdf_file_path = os.path.join(pdf_folder, base_filename + ".pdf")
        pdf.output(pdf_file_path)
        print(f"Created PDF: {pdf_file_path}")

        # ---- Generate DOCX File ----
        doc = Document()
        for line in normalized_text.splitlines():
            doc.add_paragraph(line)
        doc_file_path = os.path.join(doc_folder, base_filename + ".docx")
        doc.save(doc_file_path)
        print(f"Created DOCX: {doc_file_path}")
