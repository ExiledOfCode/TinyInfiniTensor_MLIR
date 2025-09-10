#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
  int rank = inputs[0]->getRank();
  dim = get_real_axis(_dim, rank);
  IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
  // 验证输入张量数量
  IT_ASSERT(inputs.size() >= 1, "Concat requires at least one input tensor.");

  // 获取第一个输入张量的形状和秩
  Shape dims = inputs[0]->getDims();
  auto rank = inputs[0]->getRank();

  // 验证拼接轴有效性
  IT_ASSERT(dim >= 0 && dim < rank,
            "Invalid concatenation axis: " + std::to_string(dim));

  // 检查所有输入张量的形状兼容性
  for (size_t i = 1; i < inputs.size(); ++i) {
    Shape other_dims = inputs[i]->getDims();
    // 验证秩相同
    IT_ASSERT(other_dims.size() == dims.size(),
              "All input tensors must have the same rank. Tensor " +
                  std::to_string(i) + " has shape " + vecToString(other_dims) +
                  ", expected rank " + std::to_string(rank));
    // 验证非拼接轴的形状相同
    for (int j = 0; j < rank; ++j) {
      if (j != dim) {
        IT_ASSERT(
            other_dims[j] == dims[j],
            "Input tensors must have the same shape on non-concatenation axes. "
            "Tensor " +
                std::to_string(i) + " has shape " + vecToString(other_dims) +
                ", expected " + vecToString(dims));
      }
    }
  }

  // 计算拼接轴上的总大小
  int concat_size = 0;
  for (const auto &input : inputs) {
    concat_size += input->getDims()[dim];
  }
  dims[dim] = concat_size;

  return {{dims}};
}

std::string ConcatObj::toString() const {
  std::ostringstream os;
  os << "Concat[" << getGuid() << "]";
  os << "(";
  for (auto input : inputs)
    os << vecToString(input->getDims()) << ",";
  os << "dim=" << dim << ",";
  os << "input=";
  for (auto input : inputs)
    os << input->getGuid() << ",";
  os << "output=" << outputs[0]->getGuid() << ")";
  return os.str();
}

} // namespace infini
