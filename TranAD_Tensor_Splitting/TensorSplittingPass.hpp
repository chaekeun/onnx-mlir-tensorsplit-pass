#pragma once

#include "mlir/Pass/Pass.h"

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createTensorSplittingPass();

}