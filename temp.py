import fitz  # PyMuPDF
import os

def find_pdfs_in_folder(folder_path):
    # List to store the paths of all PDF files found
    pdf_files = []
    # Walk through the directory, including subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a .pdf extension
            if file.endswith('.pdf'):
                # Construct the full path of the PDF file
                pdf_path = os.path.join(root, file)
                # Append the PDF file path to the list
                pdf_files.append(pdf_path)
    return pdf_files

def merge_pdfs(pdf_files, output_path):
    # Create a new empty PDF document
    merged_pdf = fitz.open()
    # Append each PDF file to the new document
    for pdf_file in pdf_files:
        pdf = fitz.open(pdf_file)
        merged_pdf.insertPDF(pdf)
        pdf.close()
    # Save the merged PDF document to the specified output path
    merged_pdf.save(output_path)
    merged_pdf.close()

# Example usage
folder_path = 'path/to/folder'  # Change this to the path of the folder where the PDFs are located
output_path = 'merged_output.pdf'  # Change this to the desired output path for the merged PDF

# Find all PDF files in the specified folder and its subfolders
pdf_files = find_pdfs_in_folder(folder_path)

# Merge the PDF files and save the result to the specified output path
merge_pdfs(pdf_files, output_path)

print('PDFs merged successfully.')
