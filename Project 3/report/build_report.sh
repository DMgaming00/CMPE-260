#!/usr/bin/env sh
# Rebuild both versions of the survey from survey.md (requires pandoc >= 3 and python3).
#   ieee/GNN_Survey_IEEE.tex   IEEE two-column paper (compile with pdflatex or on Overleaf)
#   GNN_Literature_Survey.docx Word version (equations become native Word equations)
#   GNN_Survey_IEEE_overleaf.zip  the IEEE paper + figures, ready to upload to Overleaf
set -e
cd "$(dirname "$0")"
python3 make_ieee.py
pandoc survey.md \
  --from markdown+tex_math_dollars \
  --resource-path=.:.. \
  --reference-doc=reference.docx \
  -o GNN_Literature_Survey.docx
echo "wrote $(pwd)/GNN_Literature_Survey.docx"
# Single-file upload for Overleaf (New Project -> Upload Project)
rm -f GNN_Survey_IEEE_overleaf.zip
(cd ieee && zip -qr ../GNN_Survey_IEEE_overleaf.zip GNN_Survey_IEEE.tex figures)
echo "wrote $(pwd)/GNN_Survey_IEEE_overleaf.zip"
