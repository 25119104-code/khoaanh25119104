#!/usr/bin/env python3
# ============================================================
# XUAT PDF TU operators/*.md
#
# Muc dich: PDF chi la BAN IN sinh ra tu .md, khong phai mot ban noi dung
# rieng. Sua .md xong chay lai script nay la PDF khop lai ngay — het canh
# hai ban lech nhau.
#
# CHAY:  python tools/xuat_pdf.py
#        (script tu tro ve thu muc goc project nen dung o dau chay cung duoc)
#
# Khong can cai them thu vien: bo chuyen doi Markdown viet san trong file,
# chi dung thu vien chuan. Phan in PDF goi Google Chrome o che do headless.
# ============================================================

import html
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

NGUON = "operators"
DICH = "PDF"
FILES = ["README", "conv2d", "relu", "maxpool2d", "flatten", "linear", "argmax"]

# Duong dan Chrome/Chromium thuong gap, theo thu tu uu tien
CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/opt/pw-browsers/chromium",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: Calibri, "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt; line-height: 1.5; color: #1A2733; margin: 0;
}
h1 { font-size: 20pt; color: #0F2A44; margin: 0 0 4pt; }
h2 { font-size: 14pt; color: #0F2A44; margin: 18pt 0 6pt;
     border-bottom: 1.5pt solid #1C7293; padding-bottom: 3pt; }
h3 { font-size: 12pt; color: #1C7293; margin: 14pt 0 5pt; }
h4 { font-size: 11pt; color: #0F2A44; margin: 12pt 0 4pt; }
p  { margin: 0 0 7pt; }
strong { color: #0F2A44; }
code {
  font-family: "Courier New", monospace; font-size: 9.5pt;
  background: #F2F6F8; padding: 0 3px; border-radius: 2px; color: #0F2A44;
}
pre {
  font-family: "Courier New", monospace; font-size: 9pt; line-height: 1.4;
  background: #F2F6F8; color: #0F2A44; padding: 8pt 10pt; margin: 8pt 0 12pt;
  white-space: pre-wrap; border-left: 2.5pt solid #C6D6DF;
}
pre code { background: none; padding: 0; font-size: inherit; }
blockquote {
  margin: 8pt 0 12pt; padding: 7pt 11pt; background: #F7FAFB;
  border-left: 2.5pt solid #1C7293; color: #44586B; font-style: italic;
}
blockquote strong { color: #1C7293; }
table {
  border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9.5pt;
  page-break-inside: auto;
}
th {
  background: #0F2A44; color: #fff; font-weight: bold;
  text-align: left; padding: 5pt 7pt; border: 0.5pt solid #0F2A44;
}
td { padding: 5pt 7pt; border: 0.5pt solid #C6D6DF; vertical-align: top; }
tbody tr:nth-child(odd) td { background: #F4F8FA; }
tr { page-break-inside: avoid; }
ul, ol { margin: 0 0 8pt; padding-left: 20pt; }
li { margin-bottom: 3pt; }
hr { border: none; border-top: 0.75pt solid #C6D6DF; margin: 14pt 0; }
.nguon {
  font-size: 8.5pt; color: #7A8B99; margin: 0 0 14pt;
  padding-bottom: 6pt; border-bottom: 0.75pt solid #E1E9EE;
}
"""


# ------------------------------------------------------------
# Bo chuyen doi Markdown -> HTML, du dung cho tap con dang dung trong
# operators/*.md: heading, bang, khoi code, danh sach, trich dan, hr,
# dam/nghieng/code inline.
# ------------------------------------------------------------
def inline(t):
    """Xu ly dam, nghieng, code inline, link. Escape HTML truoc."""
    # Tach khoi `code` ra truoc de khong bi cac luat khac an vao
    phan = re.split(r"(`[^`]*`)", t)
    ra = []
    for i, p in enumerate(phan):
        if i % 2 == 1:                                   # nam trong dau `
            ra.append("<code>" + html.escape(p[1:-1]) + "</code>")
            continue
        p = html.escape(p)
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', p)
        p = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", p)
        p = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", p)
        ra.append(p)
    return "".join(ra)


def hang_bang(dong):
    """Tach mot dong bang thanh danh sach o."""
    d = dong.strip()
    if d.startswith("|"):
        d = d[1:]
    if d.endswith("|"):
        d = d[:-1]
    return [o.strip() for o in d.split("|")]


def md2html(md):
    dong = md.split("\n")
    ra = []
    i = 0
    n = len(dong)
    while i < n:
        d = dong[i]
        s = d.strip()

        # --- khoi code ```
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not dong[i].strip().startswith("```"):
                buf.append(dong[i])
                i += 1
            i += 1
            ra.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # --- bang: dong hien tai va dong sau la dong ngan cach ---|---
        if s.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", dong[i + 1]):
            dau = hang_bang(s)
            i += 2
            than = []
            while i < n and dong[i].strip().startswith("|"):
                than.append(hang_bang(dong[i]))
                i += 1
            t = ["<table><thead><tr>"]
            t += ["<th>" + inline(o) + "</th>" for o in dau]
            t.append("</tr></thead><tbody>")
            for h in than:
                t.append("<tr>" + "".join("<td>" + inline(o) + "</td>" for o in h) + "</tr>")
            t.append("</tbody></table>")
            ra.append("".join(t))
            continue

        # --- duong ke ngang
        if re.match(r"^\s*(-{3,}|\*{3,})\s*$", d):
            ra.append("<hr>")
            i += 1
            continue

        # --- heading
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            c = len(m.group(1))
            ra.append(f"<h{c}>{inline(m.group(2))}</h{c}>")
            i += 1
            continue

        # --- trich dan >
        if s.startswith(">"):
            buf = []
            while i < n and dong[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", dong[i]))
                i += 1
            ra.append("<blockquote>" + md2html("\n".join(buf)) + "</blockquote>")
            continue

        # --- danh sach (co so hoac gach dau dong)
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", d)
        if m:
            co_so = bool(re.match(r"^\d+\.$", m.group(2)))
            the = "ol" if co_so else "ul"
            muc = []
            while i < n:
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", dong[i])
                if not mm:
                    # dong noi tiep cua muc truoc (thut le, khong rong)
                    if muc and dong[i].strip() and dong[i].startswith(("   ", "\t")):
                        muc[-1] += " " + dong[i].strip()
                        i += 1
                        continue
                    break
                muc.append(mm.group(3))
                i += 1
            ra.append(f"<{the}>" + "".join("<li>" + inline(x) + "</li>" for x in muc) + f"</{the}>")
            continue

        # --- dong trong
        if not s:
            i += 1
            continue

        # --- doan van: gom cac dong lien tiep
        buf = [s]
        i += 1
        while i < n:
            t = dong[i].strip()
            if (not t or t.startswith(("|", ">", "#", "```"))
                    or re.match(r"^(\s*)([-*]|\d+\.)\s+", dong[i])
                    or re.match(r"^\s*(-{3,})\s*$", dong[i])):
                break
            buf.append(t)
            i += 1
        ra.append("<p>" + inline(" ".join(buf)) + "</p>")

    return "\n".join(ra)


def tim_chrome():
    for p in CHROME_PATHS:
        if os.path.exists(p):
            return p
    for ten in ("google-chrome", "chromium", "chromium-browser"):
        p = shutil.which(ten)
        if p:
            return p
    return None


def main():
    os.makedirs(DICH, exist_ok=True)
    chrome = tim_chrome()
    if not chrome:
        print("Khong tim thay Chrome/Chromium.")
        print("Script van sinh file .html trong PDF/html/ — mo bang trinh duyet")
        print("roi Cmd+P -> Save as PDF la duoc.")
    else:
        print("Dung trinh duyet:", chrome)

    os.makedirs(os.path.join(DICH, "html"), exist_ok=True)
    xong, loi = [], []

    for ten in FILES:
        md_path = os.path.join(NGUON, ten + ".md")
        if not os.path.exists(md_path):
            loi.append(f"{ten}: khong co {md_path}")
            continue

        md = open(md_path, encoding="utf-8").read()
        than = md2html(md)
        trang = (
            "<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'>"
            f"<title>{html.escape(ten)}</title><style>{CSS}</style></head><body>"
            f"<div class='nguon'>Sinh tu <code>operators/{html.escape(ten)}.md</code>"
            " bang <code>tools/xuat_pdf.py</code> — dung sua truc tiep file PDF nay,"
            " sua file .md roi chay lai script.</div>"
            f"{than}</body></html>"
        )
        html_path = os.path.abspath(os.path.join(DICH, "html", ten + ".html"))
        open(html_path, "w", encoding="utf-8").write(trang)

        if not chrome:
            xong.append(f"{ten}.html")
            continue

        pdf_path = os.path.abspath(os.path.join(DICH, ten + ".pdf"))
        r = subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
             "file://" + html_path],
            capture_output=True, text=True, timeout=120,
        )
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            xong.append(f"{ten}.pdf  ({os.path.getsize(pdf_path):,} byte)")
        else:
            loi.append(f"{ten}: Chrome khong xuat duoc — {r.stderr.strip()[:200]}")

    print()
    print("=" * 60)
    for x in xong:
        print("  OK  ", x)
    for x in loi:
        print("  LOI ", x)
    print("=" * 60)
    print(f"{len(xong)} file trong {DICH}/ — sinh tu {NGUON}/*.md")
    return 1 if loi else 0


if __name__ == "__main__":
    sys.exit(main())
