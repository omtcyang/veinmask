// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

// at::Tensor veinmask_cuda(const at::Tensor points, const at::Tensor distance, int upper, int lower, const at::Tensor thetas);
void veinmask_cuda(const at::Tensor &points, const at::Tensor &distance, int upper, int lower, const at::Tensor &thetas, at::Tensor &coors);

// at::Tensor veinmask(const at::Tensor points, const at::Tensor distance, int upper, int lower, const at::Tensor thetas)
//{
//   CHECK_CUDA(distance);
//   return veinmask_cuda(points, distance, upper, lower, thetas);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("veinmask", &veinmask_cuda);
}