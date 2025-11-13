#!/usr/bin/env python3
"""
Convert all slide PDFs to HTML format across all modules.
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF


def convert_pdf_to_html(pdf_path, output_path):
    """
    Convert a PDF file to HTML format.

    Args:
        pdf_path: Path to the input PDF file
        output_path: Path to the output HTML file
    """
    print(f"Converting {pdf_path} to {output_path}...")

    try:
        # Open the PDF
        doc = fitz.open(pdf_path)

        # Convert to HTML
        html_content = []
        html_content.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .page {{
            background-color: white;
            margin: 20px 0;
            padding: 40px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 4px;
            min-height: 400px;
        }}
        .page-number {{
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .page-content {{
            line-height: 1.8;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 12pt;
        }}
        .note {{
            color: #666;
            font-style: italic;
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 3px solid #ccc;
        }}
    </style>
</head>
<body>
<div class="note">
    This is a text-only version of the slides. For the full visual experience with images and formatting, please refer to the PDF version.
</div>
""".format(Path(pdf_path).stem))

        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]

            html_content.append(f'<div class="page">')
            html_content.append(f'<div class="page-number">Page {page_num + 1} of {len(doc)}</div>')
            html_content.append('<div class="page-content">')

            # Get page as plain text (no embedded data)
            page_text = page.get_text("text")
            # Escape HTML special characters
            page_text = page_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_content.append(page_text)

            html_content.append('</div>')  # page-content
            html_content.append('</div>')  # page

        html_content.append("""
</body>
</html>
""")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))

        doc.close()
        print(f"✓ Successfully converted {pdf_path}")
        return True

    except Exception as e:
        print(f"✗ Error converting {pdf_path}: {e}")
        return False


def find_slide_pdfs(root_dir='.'):
    """
    Find all PDF files in slides directories.

    Args:
        root_dir: Root directory to search from

    Returns:
        List of paths to PDF files
    """
    pdf_files = []
    root_path = Path(root_dir)

    for slides_dir in root_path.glob('*/slides'):
        for pdf_file in slides_dir.glob('*.pdf'):
            pdf_files.append(pdf_file)

    return sorted(pdf_files)


def main():
    """Main function to convert all slide PDFs to HTML."""
    print("=" * 70)
    print("Converting Slide PDFs to HTML")
    print("=" * 70)
    print()

    # Find all slide PDFs
    pdf_files = find_slide_pdfs()

    if not pdf_files:
        print("No PDF files found in slides directories.")
        return 0

    print(f"Found {len(pdf_files)} PDF file(s) to convert:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    print()

    # Convert each PDF
    success_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        # Generate output HTML path (same name, .html extension)
        output_path = pdf_path.with_suffix('.html')

        # Check if HTML already exists
        if output_path.exists():
            print(f"⊘ Skipping {pdf_path} (HTML already exists)")
            continue

        # Convert
        if convert_pdf_to_html(str(pdf_path), str(output_path)):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print()
    print("=" * 70)
    print(f"Conversion complete: {success_count} succeeded, {fail_count} failed")
    print("=" * 70)

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
