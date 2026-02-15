#!/bin/bash
set -e

echo "Installing pgvector..."

apk add --no-cache gcc git make musl-dev postgresql-dev

git clone --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector
cd /tmp/pgvector
make USE_PGXS=1 CC=gcc NO_LTO=1 install
cd /
rm -rf /tmp/pgvector

apk del gcc git make musl-dev postgresql-dev

echo "pgvector installed successfully"
