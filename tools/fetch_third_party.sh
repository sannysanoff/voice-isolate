#!/usr/bin/env bash
# Fetch the header-only third-party sources into third_party/.
# The repository does not vendor them; the build needs these exact versions.
set -euo pipefail

EIGEN_VERSION=3.4.0
XSIMD_VERSION=13.2.0
POCKETFFT_COMMIT=4d865cba24359b05b37da4e75aee24498c3b78e7

root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/third_party"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$out"

echo "Eigen $EIGEN_VERSION"
curl -sfL "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz" |
  tar xz -C "$tmp"
rm -rf "$out/Eigen"
cp -a "$tmp/eigen-$EIGEN_VERSION/Eigen" "$out/Eigen"

echo "xsimd $XSIMD_VERSION"
curl -sfL "https://github.com/xtensor-stack/xsimd/archive/refs/tags/$XSIMD_VERSION.tar.gz" |
  tar xz -C "$tmp"
rm -rf "$out/xsimd"
cp -a "$tmp/xsimd-$XSIMD_VERSION/include/xsimd" "$out/xsimd"

echo "pocketfft $POCKETFFT_COMMIT"
curl -sfL -o "$out/pocketfft_hdronly.h" \
  "https://raw.githubusercontent.com/mreineck/pocketfft/$POCKETFFT_COMMIT/pocketfft_hdronly.h"

ls -1 "$out"
