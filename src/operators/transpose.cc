#include "operators/transpose.h"

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           vector<int> permute)
    : OperatorObj(OpType::Transpose, {input}, {output}) {
  auto rank = input->getRank();
  if (permute.empty()) {
    for (size_t i = 0; i < rank; ++i) {
      transposePermute[i] = i;
    }
  } else {
    IT_ASSERT(rank == permute.size());
    transposePermute = std::move(permute);
  }
  IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs) {
  const auto A = inputs[0];
  auto input_dim = A->getDims(); // 输入张量的形状
  auto output_dim = input_dim;   // 初始化输出形状
  int rank = A->getRank();       // 输入张量的秩

  // 如果 transposePermute 为空，默认反转维度
  if (transposePermute.empty()) {
    for (int i = 0; i < rank; ++i) {
      output_dim[i] = input_dim[rank - 1 - i];
    }
  } else {
    // 确保 transposePermute 的长度等于秩
    IT_ASSERT(rank == (int)transposePermute.size(),
              "Permute size must match input rank.");

    // 根据 transposePermute 重新排列维度
    for (int i = 0; i < rank; ++i) {
      IT_ASSERT(transposePermute[i] >= 0 && transposePermute[i] < rank,
                "Invalid permute index.");
      output_dim[i] = input_dim[transposePermute[i]];
    }
  }

  // 返回包含转置后形状的 optional<vector<Shape>>
  return {{output_dim}};
}

std::string TransposeObj::toString() const {
  std::ostringstream os;
  os << type.toString() << "[" << getGuid() << "]";
  os << "(";
  os << vecToString(inputs[0]->getDims()) << ",";
  os << "input=" << inputs[0]->getGuid() << ",";
  os << "output=" << outputs[0]->getGuid() << ")";
  return os.str();
}
}; // namespace infini
