#!/usr/bin/env python3
"""
Add home button and improved slide counter to all module presentations.
"""

import re
from pathlib import Path

TEMPLATES_DIR = Path("/Users/njpinton/projects/git/CMSC173/presenter_app/templates")

# CSS to add for home button and improved counter
HOME_BUTTON_CSS = '''
        .header-left {
            display: flex;
            align-items: center;
            gap: 15px;
            flex: 1;
        }

        .home-button {
            background: #FFD700;
            color: #1B4332;
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
        }

        .home-button:hover {
            background: white;
            transform: scale(1.05);
        }

        .slide-counter-label {
            font-size: 0.85em;
            opacity: 0.9;
            display: block;
        }
'''

def add_home_button_css(html_content):
    """Add home button CSS to the style tag."""
    # Find the closing </style> tag and insert CSS before it
    style_close_pattern = r'(\s+})\s*(</style>)'
    match = re.search(style_close_pattern, html_content)

    if match:
        insert_pos = match.start(2)
        return html_content[:insert_pos] + HOME_BUTTON_CSS + html_content[insert_pos:]
    return html_content

def add_home_button_html(html_content):
    """Add home button to the presenter header."""
    # Replace the header structure
    old_header = r'<div class="presenter-header">\s*<div>\s*<h2>(Module \d+: [^<]+)</h2>\s*</div>\s*<div class="slide-counter">'

    new_header = r'''<div class="presenter-header">
        <div class="header-left">
            <a href="/" class="home-button">🏠 Home</a>
            <h2>\1</h2>
        </div>
        <div class="slide-counter">
            <span class="slide-counter-label">Slide</span>'''

    html_content = re.sub(old_header, new_header, html_content, flags=re.MULTILINE)

    return html_content

def update_slide_counter(html_content):
    """Update slide counter display format."""
    # Replace <span id="current-slide">1</span> / <span id="total-slides">1</span>
    # with version that shows label
    pattern = r'<span id="current-slide">1</span> / <span id="total-slides">1</span>'
    replacement = '<span id="current-slide">1</span> / <span id="total-slides">1</span>'

    # This is already in the right format, just ensure label is there
    return html_content

def main():
    """Update all module HTML files."""
    html_files = sorted(TEMPLATES_DIR.glob("*.html"))

    for html_file in html_files:
        print(f"Updating {html_file.name}...")

        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if already has home button
        if 'class="home-button"' in content:
            print(f"  ✓ Already has home button, skipping")
            continue

        # Add CSS
        content = add_home_button_css(content)

        # Add home button HTML
        content = add_home_button_html(content)

        # Write back
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  ✓ Updated successfully")

    print("\nAll files updated!")

if __name__ == "__main__":
    main()
