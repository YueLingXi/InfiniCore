#ifndef __AND_CUDA_H__
#define __AND_CUDA_H__

namespace op::and_op::cuda {
typedef struct AndOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a && b;
    }
} AndOp;
} // namespace op::and_op::cuda

#endif // __AND_CUDA_H__ 