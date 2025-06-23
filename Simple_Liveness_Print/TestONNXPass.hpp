#ifndef ONNX_MLIR_TESTONNXPASS_HPP
#define ONNX_MLIR_TESTONNXPASS_HPP

#include <memory> // utilities for managing dynamic memory like std::unique_ptr
#include "mlir/Pass/Pass.h" // PassWrapper class, OperationPass template

namespace onnx_mlir {
std::unique_ptr<mlir::Pass> createTestONNXPass();
} // end of namespace onnx_mlir

#endif 
