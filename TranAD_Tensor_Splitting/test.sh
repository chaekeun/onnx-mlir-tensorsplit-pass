#!/bin/bash

set -e  # 중간에 실패 시 종료

SRC_PASS_PATH="/workdir/workspace/TestPass/TensorSplittingPass.cpp"
DEST_PASS_DIR="/workdir/onnx-mlir/src/Dialect/ONNX/Transforms"
BUILD_DIR="/workdir/onnx-mlir/build"
MLIR_INPUT="/workdir/workspace/TranAD_IR/250523/tranad.onnx.mlir"
MLIR_OUTPUT="/workdir/workspace/TranAD_IR/250523/tranad-splitted.onnx.mlir"

echo "[Step 1] Copying TensorSplittingPass.cpp..."
if cp "$SRC_PASS_PATH" "$DEST_PASS_DIR"; then
  echo "✔️  Copied successfully."
else
  echo "❌ Failed to copy TensorSplittingPass.cpp." >&2
  exit 1
fi

echo "[Step 2] Building onnx-mlir..."
cd "$BUILD_DIR" || { echo "❌ Failed to cd into $BUILD_DIR"; exit 1; }
if make -j"$(nproc)"; then
  echo "✔️  Build succeeded."
else
  echo "❌ Build failed." >&2
  exit 1
fi

echo "[Step 3] Running onnx-tensor-splitting pass..."
cd "$(dirname "$MLIR_INPUT")" || exit 1
if onnx-mlir-opt --onnx-tensor-splitting "$MLIR_INPUT" -o "$MLIR_OUTPUT"; then
  echo "✔️  Pass executed successfully. Output: $MLIR_OUTPUT"
else
  echo "❌ Failed to execute onnx-tensor-splitting pass." >&2
  exit 1
fi
