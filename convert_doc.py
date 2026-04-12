import pypandoc
import sys
from pypandoc.pandoc_download import download_pandoc

def convert():
    try:
        print("Converting Markdown to Word (.docx)...")
        output = pypandoc.convert_file('TB_Guard_XAI_ArXiv_Paper.md', 'docx', outputfile='TB_Guard_XAI_ArXiv_Paper.docx')
        print(f"Successfully created TB_Guard_XAI_ArXiv_Paper.docx!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == '__main__':
    convert()
