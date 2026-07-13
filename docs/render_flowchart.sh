#!/usr/bin/env bash

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
assets="${root}/docs/src/assets"
source_file="${assets}/diagnostic-flowchart.mmd"
svg_file="${assets}/diagnostic-flowchart.svg"
pdf_file="${assets}/diagnostic-flowchart-a4.pdf"
diagram_pdf="$(mktemp --suffix=.pdf)"

trap 'rm -f "${diagram_pdf}"' EXIT

npx -y @mermaid-js/mermaid-cli@11.16.0 \
    -i "${source_file}" \
    -o "${svg_file}" \
    -b transparent

npx -y @mermaid-js/mermaid-cli@11.16.0 \
    -i "${source_file}" \
    -o "${diagram_pdf}" \
    -b white \
    -f

gs -q \
    -o "${pdf_file}" \
    -sDEVICE=pdfwrite \
    -dFIXEDMEDIA \
    -dPDFFitPage \
    -dDEVICEWIDTHPOINTS=595 \
    -dDEVICEHEIGHTPOINTS=842 \
    "${diagram_pdf}"

printf 'Rendered %s and %s\n' "${svg_file}" "${pdf_file}"
