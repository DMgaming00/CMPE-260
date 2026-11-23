#!/usr/bin/env sh
# Rebuild both versions of the survey from survey.md (requires pandoc >= 3 and python3).
#   ieee/GNN_Survey_IEEE.tex   IEEE two-column paper (compile with pdflatex or on Overleaf)
#   GNN_Literature_Survey.docx Word version (equations become native Word equations)
set -e
cd "$(dirname "$0")"
python3 make_ieee.py
pandoc survey.md \
  --from markdown+tex_math_dollars \
  --resource-path=.:.. \
  --reference-doc=reference.docx \
  -o GNN_Literature_Survey.docx
echo "wrote $(pwd)/GNN_Literature_Survey.docx"
