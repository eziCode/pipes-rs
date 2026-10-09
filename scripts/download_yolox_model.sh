#!/bin/sh
set -eu

output="${1:-models/yolox_nano.onnx}"
url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
expected="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1"
  else
    echo "sha256sum or shasum is required to verify the model." >&2
    return 1
  fi
}

if [ -f "$output" ]; then
  actual="$(checksum "$output")"
  actual="${actual%% *}"
  if [ "$actual" != "$expected" ]; then
    echo "existing model checksum mismatch: $output" >&2
    exit 1
  fi
  echo "model already exists: $output"
  exit 0
fi

mkdir -p "$(dirname "$output")"
temporary="$output.download"
trap 'rm -f "$temporary"' EXIT HUP INT TERM
curl --fail --location --retry 3 --output "$temporary" "$url"
actual="$(checksum "$temporary")"
actual="${actual%% *}"
if [ "$actual" != "$expected" ]; then
  echo "model checksum mismatch: expected $expected, received $actual" >&2
  exit 1
fi
mv "$temporary" "$output"
trap - EXIT HUP INT TERM

echo "model: $output"
echo "sha256: $expected"
