#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/ONNX/ONNXOps.hpp" // ONNX Operations
#include "src/Pass/Passes.hpp" // Pass registration

#define DEBUG_TYPE "test-onnx-pass"

using namespace mlir;


namespace onnx_mlir {

namespace {

// This pass, TestONNXPass, traverses the operations in a module and identifies 
// elementwise binary operations implementing the ONNXElementwiseBinaryOpInterface.
// For each such operation, it emits a remark with the operation's name.
// It is primarily used for debugging or analyzing ONNX models.

struct TestONNXPass : public PassWrapper<TestONNXPass, OperationPass<func::FuncOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestONNXPass)

// function when the pass is run
    void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    Liveness liveness(funcOp);

    int blockIndex = 0;
    for (Block &block : funcOp.getBody()) {
      llvm::outs() << "\nBlock #" << blockIndex++ << ":\n";

      const LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
      if (!blockInfo) continue;

      for (Operation &op : block) {
        int64_t totalBytes = 0;
        auto liveValues = blockInfo->currentlyLiveValues(&op);

        for (Value v : liveValues) {
          if (auto tensorType = v.getType().dyn_cast<RankedTensorType>()) {
            if (!tensorType.hasStaticShape()) continue;
            size_t typeSize = tensorType.getElementType().getIntOrFloatBitWidth() / 8;
            size_t numElements = 1;
            for (int64_t dim : tensorType.getShape())
              numElements *= dim;
            totalBytes += numElements * typeSize;
          }
        }

        llvm::outs() << "  Operation: " << op.getName() << ", Live Memory: " << totalBytes << " bytes\n";
      }
    }
  }
  // Argument for the pass using in the command line
  StringRef getArgument() const override { return "test-onnx-pass"; } 
  // Description of the pass for --help
  StringRef getDescription() const override { return "250528: Test ONNX pass"; } 
};

} // end of anonymous namespace

// factory function for creating an instance of the pass
std::unique_ptr<Pass> createTestONNXPass() {
  return std::make_unique<TestONNXPass>(); 
}

} // end of namespace onnx_mlir
