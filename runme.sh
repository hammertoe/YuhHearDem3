#!/usr/bin/env bash
set -euo pipefail
IDS=(
  LPtdCCZcOOM
  df7X19vi3oo
  v4eK88964Q0
  J5SQ7bIirHo
  8CheVcGJgIo
  INF0vvE7MgE
  fAzT173IRRg
  uzKDi93xaa0
  kzdvMghUNgc
  ONsxyiKjRHA
  uxh5Oyq1SAI
  VKrnJ8ulrgI
  KTu-MFFamGQ
  8JahrKBqMtw
  iyeANTWj-w8
  aSjd9O94MY0
  AVT35BrPZIA
  xKChhsGpcf4
  WcZ4ZlS5YgI
  cTn1apqwjFk
  8ZyEWl2BpDU
  yYnhpwqZheY
  PU4vmxf9GwM
  3mWD_cAFUf4
  YoIx5icYLR0
  twui16wvTck
  Fr_GaEW-ORE
  z-Fx-ujvLYw
  _MndwltYWJg
  UBQGqpv_TN4
  BWGtHgE2ZOY
  jWIVADUCoYE
  t_AdkT9q6z0
  iQZXLJ5T5yM
  LTWQaAfcATE
  p5Kvc2vfmrg
)
for vid in "${IDS[@]}"; do
  echo "=== Transcribing ${vid} ==="
  python transcribe.py \
    --video="${vid}" \
    --segment-minutes 30 \
    --overlap-minutes 1 \
    --caption-context \
    --caption-context-buffer-seconds 30 \
    --caption-context-max-chars 1200 \
    --output-file "transcription_output_${vid}.json"
done
