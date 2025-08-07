#include "and_cpu.h"

namespace op::and_op::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    // 将handle_转换为device::cpu::Handle类型
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    // 获取输出描述符的数据类型
    auto dtype = out_desc->dtype();

    // 获取输入描述符的形状
    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &output_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    // 检查数据类型是否为BOOL类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL);

    // 检查输出描述符的形状是否与输入描述符的形状相同
    CHECK_SAME_SHAPE(output_shape, a_shape, b_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

// 计算函数，根据数据类型调用相应的计算函数
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_BOOL:
        return _device_info->calculate<AndOp, bool>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::and_op::cpu 