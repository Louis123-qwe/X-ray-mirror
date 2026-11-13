#!/usr/bin/env bash
# Collects BodyParts3D .obj files listed in an ID file into a zip.
# Usage (from the XRay Mirror folder): bash make_parts_zip.sh muscle_ids.txt muscles_extra
LIST="${1:?give the ids file, e.g. muscle_ids.txt}"; OUT="${2:-parts_extra}"; SRC="${3:-.}"
[ -f "$LIST" ] || { echo "$LIST not found"; exit 1; }
total=$(grep -c '^FJ' "$LIST"); echo "IDs to find: $total"
rm -rf "$OUT"; mkdir -p "$OUT"
ZIPS=$(find "$SRC" -maxdepth 3 \( -name "*BP3D*.zip" -o -name "*bp3d*.zip" \) -not -path "*/node_modules/*" 2>/dev/null)
echo "Zips found: ${ZIPS:-none}"
pats=""; for id in $(grep '^FJ' "$LIST"); do pats="$pats */$id.obj"; done
for z in $ZIPS; do unzip -j -n -q "$z" $pats -d "$OUT" 2>/dev/null; done
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
