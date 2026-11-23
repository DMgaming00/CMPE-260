"""Convert survey.md into an IEEEtran two-column LaTeX paper (report/ieee/).

    python make_ieee.py            # needs pandoc >= 3 on PATH

survey.md stays the single source of truth for both the .docx and the IEEE
version. This script maps its structure onto IEEEtran:
  * front matter      -> \\title, \\author, abstract, IEEEkeywords
  * "## I. Heading"     -> \\section{Heading} (IEEEtran numbers sections itself)
  * "### A. Heading"    -> \\subsection{Heading}
  * "[n]" citations     -> \\cite{bn}, references -> thebibliography
  * pipe tables         -> table / table* floats with the caption above (IEEE style)
  * images              -> figure / figure* floats
Paragraph text and math are converted by pandoc. The output folder is
self-contained (figures are copied), so it can be uploaded to Overleaf as is.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'survey.md'
OUT_DIR = HERE / 'ieee'
OUT = OUT_DIR / 'GNN_Survey_IEEE.tex'
FIG_SRC = HERE.parent / 'figures'
WIDE_FIGS = {'fig_taxonomy.png', 'fig_message_passing.png', 'fig_training_curves.png', 'fig_baselines.png'}
SEP = 'QQSEPARATORQQ'

# Characters pdflatex's utf8 input cannot typeset in text mode.
UNICODE = {
    '±': r'\ensuremath{\pm}', '×': r'\ensuremath{\times}', '≥': r'\ensuremath{\geq}',
    '≤': r'\ensuremath{\leq}', '≠': r'\ensuremath{\neq}', '≈': r'\ensuremath{\approx}',
    'α': r'\ensuremath{\alpha}', 'β': r'\ensuremath{\beta}', 'λ': r'\ensuremath{\lambda}',
    'ε': r'\ensuremath{\epsilon}', 'Σ': r'\ensuremath{\Sigma}', '‖': r'\ensuremath{\Vert}',
    '→': r'\ensuremath{\rightarrow}', '·': r'\ensuremath{\cdot}', '∝': r'\ensuremath{\propto}',
    '′': "'", '≈': r'\ensuremath{\approx}',
}
TEXT_OK = set('éčćšžüòöóáíñ–—…“”‘’')  # handled by utf8 inputenc + T1 fontenc


def pandoc_many(chunks: list[str], fmt: str = 'markdown-auto_identifiers') -> list[str]:
    """Convert many markdown snippets to LaTeX with one pandoc call."""
    doc = f'\n\n{SEP}\n\n'.join(chunks)
    out = subprocess.run(['pandoc', '-f', fmt, '-t', 'latex-auto_identifiers', '--wrap=none'],
                         input=doc, capture_output=True, text=True, check=True).stdout
    parts = [p.strip() for p in out.split(SEP)]
    if len(parts) != len(chunks):
        raise RuntimeError('pandoc chunk count mismatch')
    return parts


def cite(text: str) -> str:
    return re.sub(r'\[(\d{1,2})\]', r'\\cite{b\1}', text)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip('|').split('|')]


def table_latex(title_md: str, note_md: str, header: list[str], rows: list[list[str]], label: str) -> str:
    ncol = len(header)
    cells = header + [c for r in rows for c in r]
    conv = pandoc_many([cite(c) or 'EMPTYCELL' for c in cells] + [cite(title_md) or 'EMPTYCELL', cite(note_md) or 'EMPTYCELL'])
    conv = ['' if c == 'EMPTYCELL' else c for c in conv]
    title, note = conv[-2], conv[-1]
    conv = conv[:-2]
    head, body = conv[:ncol], [conv[ncol + i * ncol: ncol + (i + 1) * ncol] for i in range(len(rows))]
    textual = sum(len(c) for c in cells) / len(cells) > 14
    if textual:
        spec = 'l' + 'X' * (ncol - 1)
        begin, end = r'\begin{tabularx}{\textwidth}{' + spec + '}', r'\end{tabularx}'
        size = r'\scriptsize'
    else:
        spec = 'l' + 'c' * (ncol - 1)
        begin, end = r'\begin{tabular}{' + spec + '}', r'\end{tabular}'
        size = r'\footnotesize'
    lines = [r'\begin{table*}[!t]', r'\centering', size, r'\renewcommand{\arraystretch}{1.15}',
             r'\caption{' + title + '}', r'\label{' + label + '}', begin, r'\hline',
             ' & '.join(r'\textbf{' + h + '}' for h in head) + r' \\', r'\hline']
    lines += [' & '.join(r) + r' \\' for r in body]
    lines += [r'\hline', end]
    if note:
        lines += [r'\par\vspace{2pt}\raggedright ' + note]
    lines += [r'\end{table*}']
    return '\n'.join(lines)


def figure_latex(caption_md: str, path: str, n: int) -> str:
    name = Path(path).name
    cap = re.sub(r'^Fig\.\s*\d+\.\s*', '', caption_md)
    cap = pandoc_many([cite(cap)])[0]
    env = 'figure*' if name in WIDE_FIGS else 'figure'
    width = r'0.85\textwidth' if env == 'figure*' else r'\columnwidth'
    return '\n'.join([rf'\begin{{{env}}}[!t]', r'\centering', rf'\includegraphics[width={width}]{{figures/{name}}}',
                      rf'\caption{{{cap}}}', rf'\label{{fig:{n}}}', rf'\end{{{env}}}'])


def fit_display_math(tex: str) -> str:
    """Stack side-by-side display equations so they fit one IEEE column."""
    def fix(m):
        body = m.group(1)
        tag = re.search(r'\\tag\{\d+\}', body)
        tagstr = tag.group(0) if tag else ''
        core = body.replace(tagstr, '').strip()
        # equation (2): break the long message-passing update after its first argument
        core = core.replace(r'\Big(h_v^{(\ell)},\ ', r'\Big(h_v^{(\ell)}, \\ ')
        parts = [p.strip() for p in re.split(r',?\s*\\qquad\s*', core) if p.strip()]
        if len(parts) == 1 and r'\\' not in core:
            return m.group(0)
        if len(parts) > 1:
            core = r' \\ '.join(parts)
        return r'\[\begin{gathered}' + core + r'\end{gathered}' + tagstr + r'\]'
    return re.sub(r'\\\[(.*?)\\\]', fix, tex, flags=re.S)


def main() -> None:
    md = SRC.read_text()
    if '@@' in md:
        raise SystemExit('survey.md still contains @@placeholders@@; fill them first')
    fm = re.match(r'---\n(.*?)\n---\n', md, re.S)
    meta = dict(re.findall(r'^(\w+):\s*"(.*)"$', fm.group(1), re.M))
    md = md[fm.end():]

    repo = re.search(r'\*\*Code repository:\*\*\s*<([^>]*)>', md).group(1)
    base, _, folder = repo.partition('/tree/main/')
    repo_latex = r'\url{' + base + '}' + (' (folder ' + folder.replace('%20', ' ') + ')' if folder else '')
    abstract = re.search(r'## Abstract\n\n(.*?)\n\n', md, re.S).group(1)
    keywords = re.search(r'\*\*Index Terms\*\*—(.*?)\n', md).group(1).rstrip('.')
    body = md[md.index('## I. Introduction'):md.index('## References')]
    refs = re.findall(r'^\[(\d+)\] (.+)$', md[md.index('## References'):], re.M)

    out_blocks, fig_n, tab_n = [], 0, 0
    lines = body.split('\n')
    i = 0
    para: list[str] = []

    def flush():
        if para:
            out_blocks.append(('md', '\n'.join(para)))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith('## '):
            flush(); out_blocks.append(('md', '# ' + re.sub(r'^## [IVXL]+\.\s*', '', ln)))
        elif ln.startswith('### '):
            flush(); out_blocks.append(('md', '## ' + re.sub(r'^### [A-Z]\.\s*', '', ln)))
        elif ln.startswith('!['):
            flush()
            m = re.match(r'!\[(.*)\]\((.*)\)', ln)
            fig_n += 1
            out_blocks.append(('tex', figure_latex(m.group(1), m.group(2), fig_n)))
        elif ln.startswith('**Table ') and i + 2 < len(lines) and lines[i + 2].startswith('|'):
            flush()
            m = re.match(r'\*\*Table [IVX]+\.\s*(.*?)\*\*\s*(.*)', ln)
            title, note = m.group(1).rstrip('.'), m.group(2)
            j = i + 2
            tbl = []
            while j < len(lines) and lines[j].startswith('|'):
                tbl.append(lines[j]); j += 1
            header, rows = split_row(tbl[0]), [split_row(r) for r in tbl[2:]]
            tab_n += 1
            out_blocks.append(('tex', table_latex(title, note, header, rows, f'tab:{tab_n}')))
            i = j
            continue
        elif ln.strip() == '':
            flush()
        else:
            para.append(ln)
        i += 1
    flush()

    md_idx = [k for k, (t, _) in enumerate(out_blocks) if t == 'md']
    converted = pandoc_many([cite(out_blocks[k][1]) for k in md_idx])
    for k, c in zip(md_idx, converted):
        out_blocks[k] = ('tex', c)
    body_tex = '\n\n'.join(c for _, c in out_blocks)
    body_tex = fit_display_math(body_tex)

    abs_tex, kw_tex = pandoc_many([cite(abstract), keywords])
    repo_tex = repo_latex
    ref_tex = pandoc_many([r for _, r in refs], fmt='markdown+autolink_bare_uris-auto_identifiers')
    bib = '\n'.join(rf'\bibitem{{b{n}}} {t}' for (n, _), t in zip(refs, ref_tex))

    title = meta['title']
    authors = meta['author'].replace(' · ', ', ')
    doc = rf"""\documentclass[conference]{{IEEEtran}}
\IEEEoverridecommandlockouts
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{url}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\def\BibTeX{{{{\rm B\kern-.05em{{\sc i\kern-.025em b}}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{{E}}\kern-.125emX}}}}

\begin{{document}}

\title{{{title}}}

\author{{\IEEEauthorblockN{{{authors}}}
\IEEEauthorblockA{{\textit{{CMPE 260 Literature Survey and Tutorial}}\\
{meta.get('affiliation', '')}\\
Code: {repo_tex}}}}}

\maketitle

\begin{{abstract}}
{abs_tex}
\end{{abstract}}

\begin{{IEEEkeywords}}
{kw_tex}
\end{{IEEEkeywords}}

{body_tex}

\begin{{thebibliography}}{{00}}
{bib}
\end{{thebibliography}}

\end{{document}}
"""
    for ch, rep in UNICODE.items():
        doc = doc.replace(ch, rep)
    leftover = sorted({c for c in doc if ord(c) > 127 and c not in TEXT_OK})
    if leftover:
        raise SystemExit(f'Unmapped non-ASCII characters: {leftover}')

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / 'figures').mkdir(exist_ok=True)
    for f in re.findall(r'figures/([\w.]+\.png)', doc):
        shutil.copy(FIG_SRC / f, OUT_DIR / 'figures' / f)
    OUT.write_text(doc)

    # Light structural checks (no TeX engine required).
    for env in ('table*', 'figure*', 'figure', 'tabularx', 'tabular', 'abstract', 'thebibliography'):
        b, e = doc.count(rf'\begin{{{env}}}'), doc.count(rf'\end{{{env}}}')
        assert b == e, f'unbalanced {env}: {b} vs {e}'
    for bad in (r'\href', r'\hypertarget', r'\pandocbounded', r'\passthrough'):
        assert bad not in doc, f'unsupported command {bad} in output'
    cited = set(re.findall(r'\\cite\{b(\d+)\}', doc))
    defined = {n for n, _ in refs}
    assert cited <= defined, f'undefined citations {cited - defined}'
    uncited = defined - cited
    print(f'wrote {OUT} ({fig_n} figures, {tab_n} tables, {len(refs)} references)')
    if uncited:
        print('WARNING: references never cited:', sorted(uncited, key=int))


if __name__ == '__main__':
    main()
