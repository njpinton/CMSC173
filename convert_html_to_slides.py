#!/usr/bin/env python3
"""
Convert old scrollable HTML presentation format to new slide-by-slide format.
Uses Module 0 as the template structure.
"""

import re
import os
from pathlib import Path

# UP Cebu color scheme
FOREST_GREEN = "#1B4332"
UP_MAROON = "#8B0000"
GOLD = "#FFD700"

MODULE_INFO = {
    1: {"title": "Parameter Estimation", "module_num": 1},
    2: {"title": "Linear Regression", "module_num": 2},
    3: {"title": "Regularization", "module_num": 3},
    4: {"title": "Exploratory Data Analysis", "module_num": 4},
    5: {"title": "Model Selection", "module_num": 5},
    6: {"title": "Cross-Validation", "module_num": 6},
    7: {"title": "PCA", "module_num": 7},
    8: {"title": "Logistic Regression", "module_num": 8},
    9: {"title": "Classification", "module_num": 9},
    10: {"title": "Kernel Methods", "module_num": 10},
    11: {"title": "Clustering", "module_num": 11},
    12: {"title": "Neural Networks", "module_num": 12},
    13: {"title": "Advanced Neural Networks", "module_num": 13},
}


def extract_slides_from_html(html_content):
    """Extract slide content from old HTML format."""
    slides = []

    # Find all slide divs
    slide_pattern = r'<div class="slide">(.*?)</div>\s*(?:<div class="section-divider">|</div>)'
    slide_matches = re.findall(slide_pattern, html_content, re.DOTALL)

    for slide_html in slide_matches:
        # Extract title (h2)
        title_match = re.search(r'<h2>(.*?)</h2>', slide_html, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Untitled Slide"

        # Get content after title
        content = slide_html
        if title_match:
            # Remove the h2 title from content
            content = slide_html[title_match.end():]

        # Clean up the content
        content = content.strip()

        slides.append({
            "title": title,
            "content": content
        })

    return slides


def create_slide_html_template(module_num, title, slides):
    """Create the new slide-by-slide HTML from extracted slides."""

    # Generate JavaScript slides array
    slides_js = "const slides = [\n"
    for i, slide in enumerate(slides):
        slides_js += f'            {{\n'
        slides_js += f'                title: "{slide["title"]}",\n'
        slides_js += f'                content: `\n'
        slides_js += f'                    {slide["content"]}\n'
        slides_js += f'                `\n'
        slides_js += f'            }}'
        if i < len(slides) - 1:
            slides_js += ",\n"
        else:
            slides_js += "\n"
    slides_js += "        ];\n"

    # Build complete HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Module {module_num}: {title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        html, body {{
            height: 100%;
            width: 100%;
        }}

        body {{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, {FOREST_GREEN} 0%, #2D6A4F 50%, {UP_MAROON} 100%);
            color: #333;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .presenter-header {{
            background: linear-gradient(135deg, {FOREST_GREEN} 0%, {UP_MAROON} 100%);
            color: white;
            padding: 15px 30px;
            border-bottom: 4px solid {GOLD};
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
            gap: 20px;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex: 1;
        }}

        .home-button {{
            background: {GOLD};
            color: {FOREST_GREEN};
            border: none;
            padding: 8px 16px;
            font-size: 0.9em;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .home-button:hover {{
            background: white;
            transform: scale(1.05);
        }}

        .presenter-header h2 {{
            font-size: 1.3em;
            font-weight: 600;
        }}

        .slide-counter {{
            font-size: 1em;
            color: {GOLD};
            font-weight: 600;
            white-space: nowrap;
            min-width: 80px;
            text-align: right;
        }}

        .slide-counter-label {{
            font-size: 0.85em;
            opacity: 0.9;
            display: block;
        }}

        .presentation-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding: 20px;
        }}

        .slide-viewer {{
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .slide-content {{
            flex: 1;
            padding: 40px;
            overflow-y: auto;
            background: white;
        }}

        .slide-content h2 {{
            color: {FOREST_GREEN};
            font-size: 2.2em;
            margin-bottom: 25px;
            border-bottom: 4px solid {GOLD};
            padding-bottom: 15px;
            font-weight: 700;
        }}

        .slide-content h3 {{
            color: {UP_MAROON};
            font-size: 1.6em;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}

        .slide-content h4 {{
            color: {FOREST_GREEN};
            font-size: 1.3em;
            margin-top: 20px;
            margin-bottom: 12px;
            font-weight: 600;
        }}

        .slide-content p {{
            margin-bottom: 15px;
            color: #555;
            font-size: 1.05em;
            line-height: 1.8;
        }}

        .slide-content ul {{
            list-style: none;
            margin: 15px 0 15px 20px;
        }}

        .slide-content ol {{
            margin: 15px 0 15px 30px;
            padding-left: 0;
        }}

        .slide-content li {{
            margin-bottom: 12px;
            color: #555;
            line-height: 1.6;
            padding-left: 15px;
            position: relative;
        }}

        .slide-content ul li:before {{
            content: "•";
            position: absolute;
            left: 0;
            color: {GOLD};
            font-weight: bold;
            font-size: 1.2em;
        }}

        .definition {{
            background: linear-gradient(120deg, #E8F5E9 0%, #F1F8F6 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid {FOREST_GREEN};
        }}

        .definition strong {{
            color: {FOREST_GREEN};
        }}

        .highlight {{
            background: linear-gradient(120deg, #FFF9E6 0%, #FFFDF2 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid {GOLD};
        }}

        .highlight strong {{
            color: {UP_MAROON};
        }}

        .warning {{
            background: linear-gradient(120deg, #FFEBEE 0%, #FFF5F7 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid {UP_MAROON};
        }}

        .warning strong {{
            color: {UP_MAROON};
        }}

        .slide-footer {{
            background: #f5f5f5;
            padding: 15px 40px;
            border-top: 2px solid #E0E0E0;
            font-size: 0.9em;
            color: #888;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}

        .footer-left {{
            color: {FOREST_GREEN};
            font-weight: 600;
        }}

        .footer-right {{
            color: {GOLD};
            font-weight: 600;
        }}

        .controls {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
            flex-shrink: 0;
        }}

        button {{
            background: linear-gradient(135deg, {FOREST_GREEN} 0%, {UP_MAROON} 100%);
            color: white;
            border: 2px solid {GOLD};
            padding: 12px 30px;
            font-size: 1em;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(27, 67, 50, 0.2);
        }}

        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(27, 67, 50, 0.3);
            color: {GOLD};
        }}

        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}

        .hidden {{
            display: none !important;
        }}

        /* Scrollbar styling */
        .slide-content::-webkit-scrollbar {{
            width: 8px;
        }}

        .slide-content::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}

        .slide-content::-webkit-scrollbar-thumb {{
            background: {FOREST_GREEN};
            border-radius: 4px;
        }}

        .slide-content::-webkit-scrollbar-thumb:hover {{
            background: {UP_MAROON};
        }}

        @media (max-width: 768px) {{
            .presenter-header {{
                flex-direction: column;
                gap: 10px;
            }}

            .slide-content {{
                padding: 25px;
            }}

            .slide-content h2 {{
                font-size: 1.8em;
            }}

            button {{
                padding: 10px 20px;
                font-size: 0.9em;
            }}
        }}
    </style>
</head>
<body>
    <div class="presenter-header">
        <div class="header-left">
            <a href="/" class="home-button">🏠 Home</a>
            <h2>Module {module_num}: {title}</h2>
        </div>
        <div class="slide-counter">
            <span class="slide-counter-label">Slide</span>
            <span id="current-slide">1</span> / <span id="total-slides">1</span>
        </div>
    </div>

    <div class="presentation-container">
        <div class="slide-viewer">
            <div class="slide-content" id="slide-content">
                <!-- Slides will be inserted here -->
            </div>
            <div class="slide-footer">
                <div class="footer-left">CMSC 173: Machine Learning</div>
                <div class="footer-right">University of the Philippines - Cebu</div>
            </div>
        </div>

        <div class="controls">
            <button id="prev-btn" onclick="previousSlide()">← Previous</button>
            <button id="next-btn" onclick="nextSlide()">Next →</button>
        </div>
    </div>

    <script>
        // Slide content data
        {slides_js}

        let currentSlideIndex = 0;

        function renderSlide() {{
            const slide = slides[currentSlideIndex];
            const contentDiv = document.getElementById('slide-content');

            contentDiv.innerHTML = `
                <h2>${{slide.title}}</h2>
                ${{slide.content}}
            `;

            // Update slide counter
            document.getElementById('current-slide').textContent = currentSlideIndex + 1;
            document.getElementById('total-slides').textContent = slides.length;

            // Update button states
            document.getElementById('prev-btn').disabled = currentSlideIndex === 0;
            document.getElementById('next-btn').disabled = currentSlideIndex === slides.length - 1;

            // Scroll to top of slide content
            contentDiv.scrollTop = 0;
        }}

        function nextSlide() {{
            if (currentSlideIndex < slides.length - 1) {{
                currentSlideIndex++;
                renderSlide();
            }}
        }}

        function previousSlide() {{
            if (currentSlideIndex > 0) {{
                currentSlideIndex--;
                renderSlide();
            }}
        }}

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight') nextSlide();
            if (e.key === 'ArrowLeft') previousSlide();
        }});

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            document.getElementById('total-slides').textContent = slides.length;
            renderSlide();
        }});
    </script>
</body>
</html>
'''

    return html


def main():
    """Convert all module HTML files."""
    templates_dir = Path("/Users/njpinton/projects/git/CMSC173/presenter_app/templates")

    # Process modules 1-13
    for module_num in range(1, 14):
        filename = f"{module_num:02d}-{MODULE_INFO[module_num]['title'].lower().replace(' ', '-')}.html"
        filepath = templates_dir / filename

        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        print(f"Processing Module {module_num}: {MODULE_INFO[module_num]['title']}...")

        # Read old HTML
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Extract slides
        slides = extract_slides_from_html(html_content)

        if not slides:
            print(f"  WARNING: No slides extracted for Module {module_num}")
            continue

        print(f"  Found {len(slides)} slides")

        # Create new HTML
        new_html = create_slide_html_template(
            module_num,
            MODULE_INFO[module_num]['title'],
            slides
        )

        # Write new HTML
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_html)

        print(f"  ✓ Converted and saved {filepath}")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
