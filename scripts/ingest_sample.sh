#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 file.pdf"
  exit 1
fi

FILE="$1"

curl -X POST "http://localhost:8000/api/ingest" -F "files=@${FILE}" -H "Accept: application/json"
