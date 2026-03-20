import os
import glob
import re

html_dir = '.'

# 1. Rename files
renames = {
    'pipeline.html': 'v1_legacy.html',
    'pgie.html': 'v2_native.html',
    'hybrid.html': 'v3_hybrid.html'
}

for old, new in renames.items():
    if os.path.exists(old):
        os.rename(old, new)
        print(f"Renamed {old} to {new}")

# Create v4
import shutil
if os.path.exists('v3_hybrid.html'):
    shutil.copy('v3_hybrid.html', 'v4_desktop.html')
    print("Created v4_desktop.html")

# 2. Update navigation block in all HTML files
new_nav_template = """  <nav>
    <a href="index.html" class="{index_active}"><span class="nav-icon">🏠</span> Overview</a>
    <a href="install.html" class="{install_active}"><span class="nav-icon">📦</span> Installation</a>
    
    <div style="margin: 20px 0 8px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); font-weight: 600;">Application Versions</div>
    <a href="v1_legacy.html" class="{v1_legacy_active}"><span class="nav-icon">🐢</span> V1: Legacy OpenCV</a>
    <a href="v2_native.html" class="{v2_native_active}"><span class="nav-icon">🎬</span> V2: Native OSD &amp; RTSP</a>
    <a href="v3_hybrid.html" class="{v3_hybrid_active}"><span class="nav-icon">🤝</span> V3: Hybrid Tracking</a>
    <a href="v4_desktop.html" class="{v4_desktop_active}"><span class="nav-icon">💻</span> V4: GTK Desktop App</a>

    <div style="margin: 20px 0 8px 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); font-weight: 600;">Technical Deep Dives</div>
    <a href="engine.html" class="{engine_active}"><span class="nav-icon">⚡</span> TensorRT Engine</a>
    <a href="tracker-logic.html" class="{tracker_logic_active}"><span class="nav-icon">🎯</span> Tracker Logic</a>
    <a href="multi-object.html" class="{multi_object_active}"><span class="nav-icon">👥</span> Multi-Object Manager</a>
  </nav>"""

all_html_files = glob.glob('*.html')

for file_path in all_html_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Determine which file this is to set the active class
    basename = os.path.basename(file_path)
    file_key = basename.replace('.html', '').replace('-', '_')
    
    format_dict = {
        'index_active': '', 'install_active': '', 'v1_legacy_active': '',
        'v2_native_active': '', 'v3_hybrid_active': '', 'v4_desktop_active': '',
        'engine_active': '', 'tracker_logic_active': '', 'multi_object_active': ''
    }
    
    active_key = f"{file_key}_active"
    if active_key in format_dict:
        format_dict[active_key] = 'active'
        
    nav_html = new_nav_template.format(**format_dict)
    
    # Clean up empty class="" attributes
    nav_html = nav_html.replace(' class=""', '')

    # Regex to replace the *first* <nav> block (the main navigation)
    pattern = re.compile(r'<div class="sidebar-section">Navigation</div>\s*<nav>.*?</nav>\s*<div class="sidebar-section"', re.DOTALL)
    new_content = pattern.sub(f'<div class="sidebar-section">Navigation</div>\n{nav_html}\n\n  <div class="sidebar-section"', content)
    
    # Alternatively, if there was NO secondary section, just replace up to next tag. Since all have a secondary section mostly, let's just use simple replacement:
    # Actually, some files have On This Page, some have Quick Links.
    # The safest is replacing exactly the navigation tags.
    pattern2 = re.compile(r'<div class="sidebar-section">Navigation</div>\s*<nav>.*?</nav>', re.DOTALL)
    new_content_fallback = pattern2.sub(f'<div class="sidebar-section">Navigation</div>\n{nav_html}', content)
    
    # Also replace any internal links referenced in the text
    new_text = new_content_fallback.replace('href="pipeline.html"', 'href="v1_legacy.html"')
    new_text = new_text.replace('href="pgie.html"', 'href="v2_native.html"')
    new_text = new_text.replace('href="hybrid.html"', 'href="v3_hybrid.html"')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)

print("Navigation restructure complete.")
