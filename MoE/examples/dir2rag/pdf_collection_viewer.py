import streamlit as st
import os
import base64
import fitz  # PyMuPDF
import io

# Helper: Compute a directory stamp based on the modification times of all PDFs.
def get_directory_stamp(directory):
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    # Sum of modification times acts as a simple stamp (changes if any file is updated)
    stamp = sum(os.path.getmtime(os.path.join(directory, f)) for f in pdf_files)
    return stamp

# Cache the list of PDFs using a stamp so the cache is invalidated if any PDF changes.
@st.cache_data
def list_pdf_files(directory, stamp):
    files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    # Sort files by modification time (most recent first)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    return files

# Cache the reading of a PDF file for I/O performance.
@st.cache_data
def read_pdf_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()

# Cache thumbnail generation for performance.
# This version returns the PNG bytes (no resizing) as in the original implementation.
@st.cache_data
def generate_thumbnail(pdf_path, zoom=0.8):
    """
    Generates a thumbnail image (PNG bytes) for the first page of the PDF.
    The zoom factor controls the resolution of the thumbnail.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        # Create a transformation matrix for the desired zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return img_bytes
    except Exception as e:
        st.error(f"Error generating thumbnail for {os.path.basename(pdf_path)}: {e}")
        return None

def show_pdf(file_bytes):
    # Encode PDF in base64 and embed in an iframe for reading.
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main Application Layout
st.title("PDF Browser and Reader")

# Preconfigured PDF directory
pdf_directory = "/Users/pepelopez/Documents/Learning/Genai/Papers"

if pdf_directory and os.path.isdir(pdf_directory):
    # Compute the directory stamp to include in the cache key
    dir_stamp = get_directory_stamp(pdf_directory)
    # List PDFs from the directory using the stamp
    pdf_files = list_pdf_files(pdf_directory, dir_stamp)
    
    # Create two tabs: one for the grid preview and one for the PDF viewer
    preview_tab, viewer_tab = st.tabs(["Grid Preview", "PDF Viewer"])
    
    with preview_tab:
        st.header("Grid Preview with Thumbnails")
        st.markdown(
            """
            <style>
            .fixed-thumb {
                height: 200px;
                object-fit: cover;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # Create a grid layout using columns (3 columns per row)
        cols = st.columns(4)
        for idx, pdf_file in enumerate(pdf_files):
            col = cols[idx % 4]
            pdf_path = os.path.join(pdf_directory, pdf_file)
            # Generate a thumbnail (PNG bytes) for the PDF
            thumbnail = generate_thumbnail(pdf_path)
            if thumbnail:
                # Display the thumbnail image
                col.markdown(
                    f'''
                    <div style="text-align:center;">
                        <img class="fixed-thumb" src="data:image/png;base64,{base64.b64encode(thumbnail).decode('utf-8')}" />
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                col.write(pdf_file)
            # Add a button for selecting the PDF file
            if col.button("Open", key=pdf_file, help=pdf_file):
                st.session_state.selected_pdf = pdf_path
                st.info("PDF selected! Switch to the 'PDF Viewer' tab to read.")
    
    with viewer_tab:
        st.header("PDF Viewer")
        selected_pdf = st.session_state.get("selected_pdf", None)
        if selected_pdf:
            st.sidebar.info(os.path.basename(selected_pdf))
            st.sidebar.image(generate_thumbnail(selected_pdf))
        else:
            st.sidebar.info("No document selected")
        
        if selected_pdf:
            # Read and display the selected PDF file
            file_bytes = read_pdf_bytes(selected_pdf)
            show_pdf(file_bytes)
        else:
            st.info("Please select a PDF from the Grid Preview tab.")
else:
    st.error("Enter a valid directory containing PDF files.")