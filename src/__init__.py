from fpdf import FPDF

# Function to generate PDF for each row of text
def create_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)  # Adds the text to the PDF with line breaks
    pdf.output(filename)

# Function to loop through text data and create PDFs for each row
def generate_pdfs(text_data):
    for i, text in enumerate(text_data, 1):
        filename = f"row_{i}.pdf"
        create_pdf(text, filename)
        print(f"Created {filename}")
