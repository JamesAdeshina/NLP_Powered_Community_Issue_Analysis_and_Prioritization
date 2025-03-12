import os
import csv
from fpdf import FPDF
from docx import Document
from docx.shared import Pt, Inches


# Function to generate a styled PDF for each row of text
def create_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    # Set left and right margins (approx 20 mm)
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.set_font("Arial", size=12)
    # Replace non-Latin1 characters to avoid UnicodeEncodeError
    text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)


# Function to generate a styled DOCX for each row of text
def create_docx(text, filename):
    doc = Document()
    # Set page margins for each section (1 inch margins)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
    # Set default style to Arial 12pt
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(12)

    # Split text into paragraphs by newline (if present)
    for para in text.split('\n'):
        doc.add_paragraph(para)

    doc.save(filename)


# Function to loop through text data and create both PDFs and DOCX files for each row (with a limit)
def generate_documents(text_data, limit=None):
    pdf_folder = "PDF"
    docx_folder = "docx"

    # Create directories if they don't exist
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
    if not os.path.exists(docx_folder):
        os.makedirs(docx_folder)

    for i, text in enumerate(text_data[:limit], 1):
        pdf_filename = os.path.join(pdf_folder, f"row_{i}.pdf")
        docx_filename = os.path.join(docx_folder, f"row_{i}.docx")
        create_pdf(text, pdf_filename)
        create_docx(text, docx_filename)
        print(f"Created {pdf_filename} and {docx_filename}")


# Function to read CSV and get the desired column of text
def read_csv(file_name, column_index):
    text_data = []
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row if necessary
        for row in csvreader:
            if len(row) > column_index:
                text_data.append(row[column_index])
    return text_data


# Example usage:
csv_file = "new.csv"  # Replace with your CSV file path
column_index = 0  # Index of the column with letter text (0-based)
text_data = read_csv(csv_file, column_index)

# Set a limit for the number of rows to process (e.g., 5)
row_limit = 15
generate_documents(text_data, limit=row_limit)
