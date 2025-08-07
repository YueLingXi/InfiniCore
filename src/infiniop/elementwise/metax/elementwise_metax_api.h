#ifndef __INFINIOP_ELEMENTWISE_METAX_API_H__
#define __INFINIOP_ELEMENTWISE_METAX_API_H__

#include "../elementwise.h"

namespace op::elementwise::metax {

class DeviceImpl final {
    // 定义一个不透明的结构体
    struct Opaque;
    // 定义一个指向不透明结构体的共享指针
    std::shared_ptr<Opaque> _opaque;

    // 构造函数，接受一个指向不透明结构体的共享指针
    DeviceImpl(std::shared_ptr<Opaque> opaque) : _opaque(std::move(opaque)) {}

public:
    // 默认析构函数
    ~DeviceImpl() = default;

    // 创建一个DeviceImpl对象的模板函数，接受任意数量参数
    template <typename... Args>
    static utils::Result<DeviceImpl *> create(Args &&...args);

    // 计算函数，接受BLOCK_SIZE、Op、Tdata、Args等参数
    template <uint32_t BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);

    // 计算函数，接受BLOCK_SIZE、Op、Tout、Tin、Args等参数，并且要求Tin的数量等于Op::num_inputs
    template <uint32_t BLOCK_SIZE, typename Op, typename Tout, typename... Tin,
              typename... Args,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);
};
} // namespace op::elementwise::metax
#define CREATE_ELEMENTWISE_METAX_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)          \
                                                                                              \
    auto info_result = op::elementwise::ElementwiseInfo::create(OUT_DESC, INPUT_DESC_VEC);    \
    CHECK_RESULT(info_result);                                                                \
    auto info = info_result.take();                                                           \
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *);       \
                                                                                              \
    auto device_impl_result = op::elementwise::metax::DeviceImpl::create(HANDLE->internal()); \
    CHECK_RESULT(device_impl_result);                                                         \
                                                                                              \
    *desc_ptr = new Descriptor(                                                               \
        DTYPE,                                                                                \
        std::move(info),                                                                      \
        std::move(device_impl_result.take()),                                                 \
        workspace_size,                                                                       \
        HANDLE->device,                                                                       \
        HANDLE->device_id);

#endif // __INFINIOP_ELEMENTWISE_METAX_API_H__
