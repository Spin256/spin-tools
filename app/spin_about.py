# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Combine shared and per-tool About information. Tabs: About, License, Additional Terms and
Notice, the last three showing the suite's LICENSE, ADDITIONAL_TERMS.md and NOTICE.md."""
import html
import json
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget, QTextBrowser, QVBoxLayout

SUITE = Path(__file__).resolve().parent.parent
LEGAL_TABS = (('License', 'LICENSE'), ('Additional Terms', 'ADDITIONAL_TERMS.md'), ('Notice', 'NOTICE.md'))


def legal_page(name):
    """A read-only page showing a suite file: Markdown rendered, plain text as is."""
    page = QTextBrowser()
    page.setOpenExternalLinks(True)
    try:
        text = (SUITE / name).read_text(encoding='utf-8-sig')
    except OSError:
        page.setPlainText(f'{name} was not found in the Spin Tools folder.')
        return page
    if name.endswith('.md'):
        page.setMarkdown(text)
    else:
        page.setPlainText(text)
    return page


def load_about(tool_folder=None):
    with (SUITE / 'about.json').open(encoding='utf-8-sig') as source:
        shared = json.load(source)
    local = {}
    if tool_folder is not None:
        with (Path(tool_folder) / 'about.json').open(encoding='utf-8-sig') as source:
            local = json.load(source)
    merged = {**shared, **local}
    for field in ('links', 'credits'):
        entries = {entry['name']: entry for entry in shared.get(field, [])}
        entries.update({entry['name']: entry for entry in local.get(field, [])})
        merged[field] = list(entries.values())
    catalog = {**shared.get('acknowledgments', {}), **local.get('acknowledgments', {})}
    merged['acknowledgments'] = [catalog[key] for key in dict.fromkeys(merged.get('dependencies', []))]
    return merged


def escape(value):
    return html.escape(str(value))


def link(label, url):
    if urlsplit(url).scheme not in ('http', 'https'):
        raise ValueError('About links must use HTTP or HTTPS.')
    return f'<a href="{escape(url)}">{escape(label)}</a>'


class AboutDialog(QDialog):
    def __init__(self, parent=None, tool_folder=None, version=None):
        super().__init__(parent)
        info = load_about(tool_folder)
        self.setWindowTitle('About ' + info['name'])
        self.resize(640, 520)
        layout = QVBoxLayout(self)
        content = f'<h2>{escape(info["name"])}</h2>'
        if version:
            content += f'<p>Version {escape(version)}</p>'
        content += f'<p>{escape(info.get("description", ""))}</p>'
        for entry in info.get('credits', []):
            content += f'<p>{escape(entry["name"])} · {escape(entry.get("role", ""))}</p>'
        content += '<p>' + ' · '.join(link(entry['name'], entry['url']) for entry in info['links']) + '</p>'
        if info['acknowledgments']:
            content += '<hr><h2>Acknowledgments</h2>'
        for entry in info['acknowledgments']:
            content += f'<h3>{escape(entry["name"])}</h3><p>{escape(entry.get("description", ""))}</p>'
            content += f'<p>{escape(entry.get("author", ""))} · {escape(entry.get("license", ""))}</p>'
            content += '<p>' + ' · '.join(link(item['name'], item['url']) for item in entry.get('links', [])) + '</p>'
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(content)
        self.tabs = QTabWidget()
        self.tabs.addTab(browser, 'About')
        for title, name in LEGAL_TABS:
            self.tabs.addTab(legal_page(name), title)
        layout.addWidget(self.tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

