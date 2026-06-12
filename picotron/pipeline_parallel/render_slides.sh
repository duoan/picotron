#!/usr/bin/env bash
# Render teaching_slides.md to HTML / PDF / PPTX.
#
# Marp's headless-Chrome export does not load local SVGs referenced by
# `![](figures/x.svg)`, so we inline each SVG as raw markup before rendering.
# The source teaching_slides.md keeps the clean `![w:N](figures/x.svg)` form,
# which also renders directly in any browser/markdown preview.
set -euo pipefail
cd "$(dirname "$0")"

SRC=teaching_slides.md
TMP=$(mktemp --suffix=.md)
trap 'rm -f "$TMP"' EXIT

# Inline figures: ![w:N](figures/x.svg) -> centered raw <svg width="N" ...>
python3 - "$SRC" "$TMP" <<'PY'
import re, sys, pathlib, base64, mimetypes
src, tmp = sys.argv[1], sys.argv[2]
text = pathlib.Path(src).read_text()
def repl(m):
    w, path = m.group(1), m.group(2)
    svg = pathlib.Path(path).read_text()
    svg = re.sub(r'<svg ', f'<svg width="{w}" ', svg, count=1)
    # markdown-it ends a raw-HTML block at the first blank line, which would
    # spill the rest of the SVG into the page as flowed text. Drop blank lines.
    svg = re.sub(r'\n\s*\n', '\n', svg).strip()
    return f'<div style="text-align:center">{svg}</div>'
text = re.sub(r'!\[w:(\d+)\]\((figures/[^)]+\.svg)\)', repl, text)
# Raster figures (png/jpg/...) are rendered from a temp md in /tmp, so relative
# paths break. Embed them as self-contained data URIs (keeps the source clean).
def repl_raster(m):
    w, path = m.group(1), m.group(2)
    mime = mimetypes.guess_type(path)[0] or 'image/png'
    b64 = base64.b64encode(pathlib.Path(path).read_bytes()).decode()
    return f'<div style="text-align:center"><img width="{w}" src="data:{mime};base64,{b64}"></div>'
text = re.sub(r'!\[w:(\d+)\]\((figures/[^)]+\.(?:png|jpe?g|gif|webp))\)', repl_raster, text)
# enable raw HTML for inline svg
text = text.replace("marp: true\n", "marp: true\nhtml: true\n", 1)
pathlib.Path(tmp).write_text(text)
PY

# Locate a Chrome for Marp. We keep it OUTSIDE the repo (a 380M binary should
# never live in the source tree) in a user cache dir; auto-install if missing.
CHROME_CACHE="${MARP_CHROME_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/marp-chrome}"
if [ -z "${CHROME_PATH:-}" ]; then
  CHROME_PATH=$(ls -d "$CHROME_CACHE"/chrome/*/chrome-linux64/chrome 2>/dev/null | head -1 || true)
fi
if [ -z "${CHROME_PATH:-}" ] || [ ! -x "${CHROME_PATH:-}" ]; then
  echo "Chrome not found; installing into $CHROME_CACHE ..."
  npx -y @puppeteer/browsers install chrome@stable --path "$CHROME_CACHE" < /dev/null
  CHROME_PATH=$(ls -d "$CHROME_CACHE"/chrome/*/chrome-linux64/chrome 2>/dev/null | head -1 || true)
fi
export CHROME_PATH PUPPETEER_SKIP_DOWNLOAD=1

for fmt in html pdf pptx; do
  npx -y @marp-team/marp-cli@latest --no-stdin --html --allow-local-files \
    --"$fmt" "$TMP" -o "teaching_slides.$fmt" < /dev/null
done
echo "rendered: teaching_slides.{html,pdf,pptx}"
