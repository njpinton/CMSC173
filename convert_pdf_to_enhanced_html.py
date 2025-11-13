#!/usr/bin/env python3
"""
Convert PDF slides to enhanced HTML presentations
Preserves original content while improving formatting and visuals
"""

import os
import subprocess
from pdf2image import convert_from_path
from PIL import Image
import io
import base64

# Enhanced HTML template with modern styling
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        html {{
            scroll-behavior: smooth;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
            border-bottom: 4px solid #5568d3;
        }}

        .header h1 {{
            font-size: 2.8em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            font-weight: 700;
        }}

        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.95;
            margin-bottom: 5px;
        }}

        .header .institution {{
            font-size: 0.95em;
            opacity: 0.85;
        }}

        .toc {{
            background: #f8f9fa;
            padding: 30px 40px;
            border-bottom: 2px solid #e9ecef;
        }}

        .toc h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
        }}

        .toc ul {{
            list-style: none;
            columns: 2;
            gap: 30px;
        }}

        .toc li {{
            margin-bottom: 10px;
            padding-left: 25px;
            position: relative;
        }}

        .toc li:before {{
            content: "▶";
            position: absolute;
            left: 0;
            color: #667eea;
        }}

        .content {{
            padding: 40px;
        }}

        .slide {{
            margin: 40px 0;
            padding: 35px;
            background: #f8f9fa;
            border-left: 6px solid #667eea;
            border-radius: 8px;
            page-break-inside: avoid;
            transition: all 0.3s ease;
        }}

        .slide:hover {{
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
        }}

        .slide h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 12px;
            font-weight: 700;
        }}

        .slide h3 {{
            color: #764ba2;
            font-size: 1.5em;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 600;
        }}

        .slide h4 {{
            color: #5568d3;
            font-size: 1.2em;
            margin-top: 20px;
            margin-bottom: 12px;
            font-weight: 600;
        }}

        .slide p {{
            margin-bottom: 15px;
            color: #555;
            font-size: 1.05em;
            line-height: 1.7;
        }}

        .slide ul {{
            list-style: none;
            margin: 15px 0;
            margin-left: 0;
        }}

        .slide ol {{
            margin: 15px 0;
            margin-left: 30px;
        }}

        .slide li {{
            margin-bottom: 12px;
            color: #555;
            line-height: 1.6;
            padding-left: 15px;
            position: relative;
        }}

        .slide ul li:before {{
            content: "•";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }}

        .slide code {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
        }}

        .slide pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            border-left: 4px solid #667eea;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.5;
        }}

        .highlight {{
            background: linear-gradient(120deg, #fff3cd 0%, #fffbea 100%);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #ffc107;
            border-radius: 8px;
        }}

        .highlight strong {{
            color: #ff6b6b;
        }}

        .note {{
            background: linear-gradient(120deg, #d1ecf1 0%, #e8f4f8 100%);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #17a2b8;
        }}

        .note strong {{
            color: #0c5460;
        }}

        .warning {{
            background: linear-gradient(120deg, #f8d7da 0%, #fce4ec 100%);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #dc3545;
        }}

        .equation {{
            background: #f0f0f0;
            padding: 25px;
            border-radius: 8px;
            margin: 25px 0;
            font-size: 1.1em;
            text-align: center;
            font-style: italic;
            border: 2px dashed #667eea;
            overflow-x: auto;
        }}

        .definition {{
            background: linear-gradient(120deg, #d4edda 0%, #e8f5e9 100%);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #28a745;
        }}

        .definition strong {{
            color: #155724;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border-radius: 8px;
            overflow: hidden;
        }}

        table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            font-weight: 600;
            text-align: left;
        }}

        table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}

        table tr:hover {{
            background: #f8f9fa;
        }}

        .slide-image {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 25px 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            border: 1px solid #e9ecef;
        }}

        .image-caption {{
            text-align: center;
            color: #666;
            font-size: 0.95em;
            margin-top: -15px;
            margin-bottom: 20px;
            font-style: italic;
        }}

        .slide-number {{
            text-align: right;
            color: #aaa;
            font-size: 0.9em;
            margin-top: 25px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }}

        .footer {{
            background: linear-gradient(to right, #f8f9fa, #f0f1f3);
            padding: 30px 40px;
            text-align: center;
            color: #666;
            border-top: 2px solid #e9ecef;
            font-size: 0.95em;
        }}

        .footer p {{
            margin: 5px 0;
        }}

        .nav {{
            display: flex;
            justify-content: space-between;
            padding: 20px 40px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }}

        .nav a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }}

        .nav a:hover {{
            color: #764ba2;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}

            .slide {{
                padding: 20px;
            }}

            .toc ul {{
                columns: 1;
            }}
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                max-width: 100%;
                border-radius: 0;
            }}
            .nav {{
                display: none;
            }}
        }}

        /* Animation */
        .slide {{
            animation: slideIn 0.5s ease-out;
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p class="subtitle">{subtitle}</p>
            <p class="institution">CMSC 173: Machine Learning | University of the Philippines - Cebu</p>
        </div>

        <div class="content">
            {content}
        </div>

        <div class="footer">
            <p><strong>{title}</strong></p>
            <p>Generated: {date}</p>
        </div>
    </div>
</body>
</html>
"""

def convert_pdf_to_enhanced_html(pdf_path, output_html_path, module_num, module_name):
    """Convert a PDF file to an enhanced HTML presentation"""

    try:
        # Convert PDF to images
        print(f"Converting Module {module_num}: {module_name}...", end=" ", flush=True)

        images = convert_from_path(pdf_path, dpi=150)

        # Create HTML content with images
        content_html = ""

        for page_num, image in enumerate(images, 1):
            # Convert PIL image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Add slide with image
            content_html += f"""
    <div class="slide">
        <img src="data:image/png;base64,{img_base64}" alt="Slide {page_num}" class="slide-image">
        <div class="slide-number">Slide {page_num}/{len(images)}</div>
    </div>
"""

        # Generate final HTML
        from datetime import datetime
        html_content = HTML_TEMPLATE.format(
            title=module_name,
            subtitle=f"Module {module_num}",
            content=content_html,
            date=datetime.now().strftime("%B %d, %Y")
        )

        # Write HTML file
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ {len(images)} pages")
        return True

    except Exception as e:
        print(f"✗ Error: {str(e)[:40]}")
        return False

# Module configurations
modules = {
    5: ("05 - Model Selection", "model_selection_slides.pdf"),
    6: ("06 - Cross Validation", "cross_validation_slides.pdf"),
    7: ("07 - PCA", "slides.pdf"),
    9: ("09 - Classification", "slides.pdf"),
    10: ("10 - Kernel Methods", "kernel_methods_slides.pdf"),
    11: ("11 - Clustering", "clustering_slides.pdf"),
    12: ("12 - Neural Networks", "neural_networks_slides.pdf"),
    13: ("13 - Advanced Neural Networks", "slides.pdf"),
}

print("CONVERTING PDFs TO ENHANCED HTML PRESENTATIONS\n")
print("=" * 70)

converted_count = 0
failed_count = 0

for module_num, (module_folder, pdf_filename) in modules.items():
    pdf_path = os.path.join(module_folder, "slides", pdf_filename)
    html_output = f"presenter_app/templates/module_{module_num:02d}.html"

    if not os.path.exists(pdf_path):
        print(f"✗ Module {module_num}: PDF not found at {pdf_path}")
        failed_count += 1
        continue

    if convert_pdf_to_enhanced_html(pdf_path, html_output, module_num, module_folder.split(' - ')[1]):
        converted_count += 1
    else:
        failed_count += 1

print("\n" + "=" * 70)
print(f"✅ Conversion complete: {converted_count} modules converted, {failed_count} failed")
print(f"HTML files saved to: presenter_app/templates/module_0X.html")
