#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir{

namespace {

struct TensorSplittingPass
    // This pass is executed on a function operation (func::FuncOp) 
    : public PassWrapper<TensorSplittingPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorSplittingPass)

  StringRef getArgument() const override { return "onnx-tensor-splitting"; }

  StringRef getDescription() const override {
    return "250528 00:25 :: Reduces memory bottlenecks by splitting tensors";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    Liveness &liveness = getAnalysis<Liveness>();
    
    // 마지막에 지울 operation들
    SmallVector<Operation *> opsToErase;

    // Step1: find SoftmaxOp with max live memory usage
    Operation *peakOp = nullptr;
    int64_t maxBytes = -1;

    for (Block &block : funcOp.getBody()) {
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

        llvm::outs() << "Operation: " << op.getName()
                    << ", Live Values Memory Usage: " << totalBytes << " bytes\n";

        if (totalBytes > maxBytes) {
          maxBytes = totalBytes;
          peakOp = &op;
        }
      }
    }

    if (!peakOp || !llvm::isa<ONNXSoftmaxOp>(peakOp)){
      llvm::outs() << "Peak op is not Softmax. Skipping tensor splitting.\n";
      return;
    }

    auto peakSoftmaxOp = llvm::cast<ONNXSoftmaxOp>(peakOp);
    llvm::outs() << "Peak SoftmaxOp found: " << peakSoftmaxOp->getName()
                 << " with live memory usage: " << maxBytes << " bytes\n";
    


    // Step2: Decompose the peak SoftmaxOp if it is optimizable
    // peakSoftmaxOp의 operand를 확인했을때 그게 matmul이면 
    // 이전 연산이 MatMul이고 다음 연산도 MatMul인 경우에만 최적화

    Operation *prevOp = peakSoftmaxOp->getOperand(0).getDefiningOp();
    Operation *nextOp = nullptr;
    for (auto &use : peakSoftmaxOp.getResult().getUses()) {
      if (auto userOp = use.getOwner()) {
        llvm::outs() << "Softmax output is used by: " << userOp->getName() << "\n";
        nextOp = userOp;
        break;
      }
    }
    if (!prevOp || !nextOp) {
      llvm::outs() << "No previous or next operation found for SoftmaxOp.\n";
      return;
    }

    if (!llvm::isa<ONNXMatMulOp>(prevOp) || !llvm::isa<ONNXMatMulOp>(nextOp)) {
      llvm::outs() << "Previous or next operation is not MatMul.\n";
      return;
    }

    // Step3: optimize the structure
    Location loc = peakSoftmaxOp.getLoc();
    builder.setInsertionPoint(peakSoftmaxOp);

    Value input = prevOp->getOperand(0);     // Q
    Value weight1 = prevOp->getOperand(1);   // K^T
    Value weight2 = nextOp->getOperand(1);   // V

    // 1. Create [5, 5] constant tensor
    auto int64Type = builder.getIntegerType(64);
    auto sint64Type = builder.getIntegerType(64, true);
    RankedTensorType splitSizeType = RankedTensorType::get({2}, int64Type);
    DenseElementsAttr splitAttr = DenseElementsAttr::get(splitSizeType, {5ll, 5ll});
    auto splitConst = builder.create<ONNXConstantOp>(loc, Attribute(), splitAttr);
    Value splitSizes = splitConst.getResult();

    // 2. Create SplitOp with static output types
    auto axisAttr = IntegerAttr::get(sint64Type, 1);

    SmallVector<Type, 2> resultTypes = {
      RankedTensorType::get({250, 5, 2}, builder.getF32Type()),
      RankedTensorType::get({250, 5, 2}, builder.getF32Type())
    };

    auto splitOp = builder.create<ONNXSplitOp>(
        loc, resultTypes, input, splitSizes, axisAttr, nullptr);


    llvm::outs() << "SplitOp num results: " << splitOp->getNumResults() << "\n";
    for (auto res : splitOp->getResults())
      llvm::outs() << "  result type: " << res.getType() << "\n";

    if (splitOp->getNumResults() < 2) {
      peakSoftmaxOp.emitError("ONNXSplitOp produced fewer than 2 results!");
      return signalPassFailure();
    }

    Value a1 = splitOp.getResult(0);
    Value a2 = splitOp.getResult(1);

    // 4. Define pipeline
    auto slicePipeline = [&](Value a) -> Value {
      auto mm1 = builder.create<ONNXMatMulOp>(loc,
          RankedTensorType::get({250, 5, 10}, builder.getF32Type()), a, weight1);
      auto sm = builder.create<ONNXSoftmaxOp>(loc,
          RankedTensorType::get({250, 5, 10}, builder.getF32Type()), mm1);
      sm->setAttr("axis", builder.getIntegerAttr(sint64Type, -1));
      auto mm2 = builder.create<ONNXMatMulOp>(loc,
          RankedTensorType::get({250, 5, 2}, builder.getF32Type()), sm, weight2);
      return mm2;
    };

    Value d1 = slicePipeline(a1);
    Value d2 = slicePipeline(a2);

    // 5. Concat 결과 병합
    auto concat = builder.create<ONNXConcatOp>(loc,
        nextOp->getResult(0).getType(), ValueRange{d1, d2});
    concat->setAttr("axis", axisAttr);

    // 연결만 바꿔줌
    nextOp->getResult(0).replaceAllUsesWith(concat);

    // 나중에 지우도록 저장
    opsToErase.push_back(prevOp);
    opsToErase.push_back(peakSoftmaxOp);
    opsToErase.push_back(nextOp);

    llvm::outs() << "Successfully decomposed SoftmaxOp into MatMul and Concat.\n";

    
    // 마지막에 안전하게 삭제
    for (Operation *op : opsToErase) {
      if (!op) continue;
      if (llvm::all_of(op->getResults(), [](Value v) { return v.use_empty(); }))
        op->erase();
    }
  }


private:

  // bool isOptimizable(Operation *peakOp){
  //   // check if the peak operation is Softmax
  //   if (llvm::isa<ONNXSoftmaxOp>(peakOp)) {
  //     // Check if the next operation is a Matmul
  //     auto prevOp = peakOp->getPrevNode();
  //     auto nextOp = peakOp->getNextNode();

  //     if (nextOp && llvm::isa<ONNXMatMulOp>(nextOp)) {
  //       // check if the previous operation is Matmul
  //       if (prevOp && llvm::isa<ONNXMatMulOp>(prevOp)) {
  //         return true;  
  //       } else {
  //         return false;  
  //       }
  //     } else{
  //       return false;
  //     }
  //   } else {
  //     return false;  
  //   }
  // }

// bool isDepthwiseConv(Operation *op) {
//   // 연산이 ONNXConvOp인지 확인
//   auto convOp = llvm::dyn_cast<ONNXConvOp>(op);
//   if (!convOp)
//     return false;

//   // W(weight/kernel) 텐서를 가져옴
//   Value kernel = convOp.W();

//   // 커널 타입 확인
//   if (auto kernelType = kernel.getType().dyn_cast<TensorType>()) {
//     ArrayRef<int64_t> shape = kernelType.getShape();

//     int64_t channelsPerGroup = shape[1];

//     // 'group' attribute 값 추출
//     auto groupAttr = convOp->getAttrOfType<IntegerAttr>("group");
//     int64_t group = groupAttr.getInt();

//     // input channel 수(C) 추출 (tensor shape: [N, C, H, W])
//     auto inputType = convOp.X().getType().dyn_cast<TensorType>();
//     int64_t C = inputType.getShape()[1];

//     // group과 input channel 수가 같고 weight의 두번째 차원(C/group)이 1이면
//     // depthwise conv
//     if ((group == C) && channelsPerGroup == 1)
//       return true;
//   }

//   return false;
// }

// bool isIntermediateLargerThanDepthwise(Operation *pwConvOp) {
//   // pointwise conv인지 확인
//   if (!isPointwiseConv(pwConvOp))
//     return false;

//   auto pwConv = llvm::dyn_cast<ONNXConvOp>(pwConvOp);
//   if (!pwConv)
//     return false;

//   // pointwise conv의 결과 (intermediate tensor)
//   Value intermediateTensor = pwConv.getResult();

//   // intermediate tensor의 사용자들을 확인
//   Operation *nextOp = nullptr;

//   for (auto user : intermediateTensor.getUsers()) {
//     // clip op가 있는 경우
//     if (auto clipOp = llvm::dyn_cast<ONNXClipOp>(user)) {
//       // clip의 결과를 사용하는 다음 연산 찾기
//       Value clipResult = clipOp.getResult();
//       for (auto clipUser : clipResult.getUsers()) {
//         if (isDepthwiseConv(clipUser)) {
//           nextOp = clipUser;
//           break;
//         }
//       }
//     }
//     // 직접 depthwise conv로 연결되는 경우
//     else if (isDepthwiseConv(user)) {
//       nextOp = user;
//       break;
//     }
//   }

//   if (!nextOp)
//     return false;

//   auto dwConv = llvm::dyn_cast<ONNXConvOp>(nextOp);
//   if (!dwConv)
//     return false;

//   // tensor 타입 가져오기
//   auto intermediateType = intermediateTensor.getType().dyn_cast<TensorType>();
//   auto dwResultType = dwConv.getResult().getType().dyn_cast<TensorType>();

//   if (!intermediateType || !dwResultType)
//     return false;

//   // getTensorSize 함수 활용해서 크기 비교
//   int64_t intermediateSize = getTensorSize(intermediateType);
//   int64_t dwResultSize = getTensorSize(dwResultType);

//   return intermediateSize > dwResultSize;
// }

// --- Core Decomposition Logic (Highly Op-Specific) ---
// This section requires significant ONNX-specific knowledge and careful
// implementation. The examples below are very conceptual.
// bool decomposeONNXOperation(Operation *op, OpBuilder &builder) {
//   builder.setInsertionPoint(op);

//   // Example: Conceptual tiling for ONNXMatMulOp
//   if (auto matmulOp = llvm::dyn_cast<ONNXMatMulOp>(op)) {
//     // THIS IS A PLACEHOLDER for a complex transformation.
//     // Real tiling involves:
//     // 1. Determining tile sizes.
//     // 2. Creating ONNXSliceOp for inputs A and B.
//     // 3. Creating loops (if not possible to stay purely in ONNX for this, then
//     // this pass is limited).
//     //    Or, unrolling into multiple smaller ONNXMatMulOps.
//     // 4. Accumulating/concatenating results using ONNXReduceSumOp/ONNXConcatOp.
//     //
//     // Example of splitting A and performing two MatMuls, then concatenating:
//     // (Highly simplified, assumes A can be split along the M dimension, B is
//     // broadcast or compatible)
//     Value A = matmulOp.A();
//     Value B = matmulOp.B();
//     Location loc = matmulOp.getLoc();
//     auto aType = A.getType().dyn_cast<TensorType>();
//     auto origResultType = matmulOp.Y().getType().dyn_cast<TensorType>();

//     if (!aType || !aType.hasRank() || aType.getRank() < 1 || !origResultType) {
//       llvm::errs() << "MatMul decomposition: Input A or original result type "
//                       "unsuitable for simple split.\n";
//       return false;
//     }

//     // Try to split the 0-th dimension of A if it's large enough.
//     int64_t M = aType.getShape()[0];
//     if (ShapedType::isDynamic(M) ||
//         M < 4) { // Need at least 4 to split into 2 reasonable parts.
//       llvm::errs() << "MatMul decomposition: Dimension M of A is dynamic or "
//                       "too small to split.\n";
//       return false;
//     }
//     int64_t M_half1 = M / 2;
//     int64_t M_half2 = M - M_half1;

    // Create ONNXSliceOp for A_part1
    // starts = [0, 0, ...], ends = [M_half1, dim1_size, dim2_size, ...], axes =
    // [0, 1, 2, ...] This requires careful setup of attributes for ONNXSliceOp.
    // The ONNXSliceOp builder needs these as I64ArrayAttr or equivalent.

    // PSEUDOCODE for slicing and dicing:
    /*
    Value aStarts1 = builder.create<ONNXConstantOp>(loc,
    DenseElementsAttr::get(RankedTensorType::get({aType.getRank()},
    builder.getI64Type()), ArrayRef<int64_t>(... build starts1 ...)); Value
    aEnds1 = builder.create<ONNXConstantOp>(loc, ..., ArrayRef<int64_t>(...
    build ends1 ...)); Value axes = builder.create<ONNXConstantOp>(loc, ...,
    ArrayRef<int64_t>(... build axes 0 to rank-1 ...));

    // Output type for slice_A1
    SmallVector<int64_t> sliceA1Shape = llvm::to_vector(aType.getShape());
    sliceA1Shape[0] = M_half1;
    auto sliceA1Type = RankedTensorType::get(sliceA1Shape,
    aType.getElementType()); Value slice_A1 = builder.create<ONNXSliceOp>(loc,
    sliceA1Type, A, aStarts1, aEnds1, axes, nullptr); // Assuming no steps

    // ... similarly for slice_A2 using M_half2 ...

    // Output types for partial MatMuls
    SmallVector<int64_t> partialResultShape =
    llvm::to_vector(origResultType.getShape()); partialResultShape[0] = M_half1;
    // Assuming result dimension corresponds to A's 0th dim auto
    partialMatMul1Type = RankedTensorType::get(partialResultShape,
    origResultType.getElementType()); Value partial_Y1 =
    builder.create<ONNXMatMulOp>(loc, partialMatMul1Type, slice_A1, B);

    // ... similarly for partial_Y2 ...

    // Concatenate results
    // The 'axis' for ONNXConcatOp would typically be the dimension that was
    split (e.g., 0) Value final_Y = builder.create<ONNXConcatOp>(loc,
    origResultType, ValueRange{partial_Y1, partial_Y2},
    builder.getI64IntegerAttr(0));

    matmulOp.Y().replaceAllUsesWith(final_Y);
    matmulOp.erase();
    return true;
    */
  //   llvm::errs() << "ONNXMatMulOp decomposition placeholder triggered. Actual "
  //                   "implementation needed.\n";
  //   return false; // Placeholder
  // }

  // Add decomposition rules for ONNXConvOp, ONNXGemmOp, etc.
  // These would involve creating ONNXSliceOp, ONNXPadOp, smaller Conv/Gemm ops,
  // and ONNXConcatOp.

  // llvm::errs() << "No decomposition rule implemented for op: " << op->getName()
  //              << "\n";
  // return false;
};

}

std::unique_ptr<Pass> createTensorSplittingPass() {
  return std::make_unique<TensorSplittingPass>();
}
}
