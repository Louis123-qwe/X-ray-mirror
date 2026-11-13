#!/usr/bin/env bash
# Needs torso_ids.txt next to this script.
# Usage (from the XRay Mirror folder): bash make_torso_zip.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-.}"; OUT=torso_extra; LIST="$HERE/torso_ids.txt"
[ -f "$LIST" ] || { echo "torso_ids.txt not found next to the script"; exit 1; }
total=$(grep -c '^FJ' "$LIST"); echo "IDs to find: $total"
rm -rf "$OUT"; mkdir -p "$OUT"
ZIPS=$(find "$SRC" -maxdepth 3 \( -name "*BP3D*.zip" -o -name "*bp3d*.zip" \) -not -path "*/node_modules/*" 2>/dev/null)
echo "Zips found: ${ZIPS:-none}"
pats=""; for id in $(grep '^FJ' "$LIST"); do pats="$pats */$id.obj"; done
for z in $ZIPS; do unzip -j -n -q "$z" $pats -d "$OUT" 2>/dev/null; done
# anything still missing: look for loose extracted files
found=0; miss=""
for id in $(grep '^FJ' "$LIST"); do
  if [ ! -f "$OUT/$id.obj" ]; then
    f=$(find "$SRC" -name "$id.obj" -not -path "*/node_modules/*" -not -path "*/$OUT/*" 2>/dev/null | head -1)
    [ -n "$f" ] && cp "$f" "$OUT/"
  fi
  if [ -f "$OUT/$id.obj" ]; then found=$((found+1)); else miss="$miss $id"; fi
done
echo "Found $found of $total files"; [ -n "$miss" ] && echo "Missing:$miss"
rm -f $OUT.zip; zip -qr $OUT.zip $OUT && echo "Created $OUT.zip ($(du -h $OUT.zip | cut -f1))"
cp $OUT.zip /sdcard/Download/ 2>/dev/null && echo "Copied to Downloads"
