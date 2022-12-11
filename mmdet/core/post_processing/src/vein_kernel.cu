#include "util.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>

#define index(y, x) (y * W + x)
#define perline (3 * line + 1) * 2
#define channels (3 * line + 1) * 2 * line
#define indexchannel(y, x, c) (c * H * W + y * W + x)

// __device__ void get_tailcoors(const float *vein, int *head, int curNum, int nxtNum, float *res, int x, int y, int H, int W,
//                               int num_tails, int is_deep, int upper, int lower, const float *thetas, int line, int root_x = 0, int root_y = 0, int root_curNum = 0,
//                               int curIndex = 0);

__device__ double point_distance(double p1_1, double p1_2, double p2_1, double p2_2)
{
  return pow((p1_1 - p2_1) * (p1_1 - p2_1) + (p1_2 - p2_2) * (p1_2 - p2_2), 0.5);
}

__device__ double point_phi(double start_1, double start_2, double end_1, double end_2)
{
  return atan2(end_2 - start_2, end_1 - start_1) * 180.0 / M_PI;
}

__device__ int check(int x, int y, int h, int w)
{
  if (x < 0 || y < 0 || x >= w || y >= h)
  {
    return 0;
  }
  return 1;
}

__device__ int clip(int x, int upper)
{
  if (x < 0)
    return 0;
  if (x > upper)
    return upper;
  return x;
}

__device__ int judge_deep(
    double thetasPhiDeep,
    double thetasPhiCur,
    double thetasPhiNxt,
    float vein_length,
    float vein_length1,
    int distance_threshold)
{
  if (thetasPhiCur > 0 && thetasPhiNxt < 0)
  {
    if (thetasPhiDeep >= 0)
      thetasPhiNxt += 360;
    else
      thetasPhiCur -= 360;
  }
  if ((thetasPhiDeep > thetasPhiCur) && (thetasPhiDeep < thetasPhiNxt) && (vein_length > distance_threshold) && (vein_length1 > distance_threshold))
    return 1;
  return 0;
}

__device__ void get_headcoor(const float *vein, int x, int y, int *head, const float *thetas, int H, int W, int line, int start)
{
  int iter = start;
  for (int i = 0; i < line; i++)
  {
    float theta = thetas[i];
    float dis = vein[indexchannel(y, x, i)];
    int h_x = int(x + cos(theta) * dis);
    int h_y = int(y + sin(theta) * dis);
    head[iter] = h_x;
    head[iter + 1] = h_y;
    iter += 2;
  }
}

__device__ int get_indexdeeps(int start_x, int start_y, int end_x, int end_y, int centerpoint_x, int centerpoint_y, const float *thetas, int line, int curNum, int *indexlist)
{
  int angle_scale = 10;
  double curPhi = atan2(start_y - centerpoint_y, start_x - centerpoint_x) * 180 / M_PI;
  double nxtPhi = atan2(end_y - centerpoint_y, end_x - centerpoint_x) * 180 / M_PI;
  //当 curPhi 到 nxtPhi 之间的顺时针夹角小于 angle_range 度，则没有必要进行deepsearch
  double phiDelta = 0;
  if (nxtPhi >= curPhi)
    phiDelta = nxtPhi - curPhi;
  else
    phiDelta = (180 - curPhi) + (nxtPhi - (-180));
  if (phiDelta < 720.0 / line)
    return 0;
  //  # ！bug！对深度搜索的角度范围进行缩小，避免深度搜索时穿过有效角度范围导致轮廓点分散化！bug！
  curPhi += angle_scale;
  nxtPhi -= angle_scale;
  if (curPhi >= 180)
    curPhi -= 360;
  if (nxtPhi <= -180)
    nxtPhi += 360;
  //   # ！bug！对深度搜索的角度范围进行缩小，避免深度搜索时穿过有效角度范围导致轮廓点分散化！bug！
  //   # 后加------------------------------------------
  if (nxtPhi >= curPhi)
    phiDelta = nxtPhi - curPhi;
  else
    phiDelta = (180 - curPhi) + (nxtPhi - (-180));
  if (phiDelta < 720.0 / line)
    return 0;
  //   # --------------------------------------------
  // # 计算有效角度范围索引序列
  int curIdx = 0;
  int nxtIdx = line - 1;

  curPhi = curPhi * M_PI / 180.0;
  nxtPhi = nxtPhi * M_PI / 180.0;

  for (int i = 0; i < line; i++)
  {
    if (thetas[i] > curPhi)
    {
      curIdx = i;
      break;
    }
  }
  for (int i = line - 1; i >= 0; i--)
  {
    if (thetas[i] < nxtPhi)
    {
      nxtIdx = i;
      break;
    }
  }
  int start = threadIdx.x * line * line + curNum * line;
  if (curIdx <= nxtIdx)
  {
    int count = nxtIdx + 1 - curIdx;
    if (count == line)
      return 0;

    for (int i = curIdx; i <= nxtIdx; i++)
      indexlist[start + i - curIdx] = i;
    return count;
  }
  else
  {
    int count = line - curIdx + nxtIdx + 1;
    if (count == line)
      return 0;

    for (int i = curIdx; i < line; i++)
      indexlist[start + i - curIdx] = i;
    for (int i = 0; i <= nxtIdx; i++)
      indexlist[start + line - curIdx + i] = i;

    return count;
  }
}

__device__ void get_deepcoors2(const float *vein, int curNum, int nxtNum, float *res, int x, int y, int H, int W, int num_tails,
                               int upper, int lower, const float *thetas, int line, int root_x, int root_y, int root_curNum, int curIndex, int *deepHead)
{
  int start = (threadIdx.x * line + root_curNum) * line * 2;
  res[index(root_y, root_x) * channels + perline * root_curNum + 2 + 6 * curIndex] = deepHead[start + curNum * 2];
  res[index(root_y, root_x) * channels + perline * root_curNum + 3 + 6 * curIndex] = deepHead[start + curNum * 2 + 1];
  //   printf("\n%f %f 1\n", res[index(root_y, root_x) * channels + perline * root_curNum + 2 + 6 * curIndex], res[index(root_y, root_x) * channels + perline * root_curNum + 3 + 6 * curIndex]);
  //   printf("\n%f\n", point_distance(deepHead[start + curNum * 2], deepHead[start + curNum * 2 + 1], deepHead[start + nxtNum * 2], deepHead[start + nxtNum * 2 + 1]));
  // printf("\n%d %d %d %d\n", deepHead[start + curNum * 2], deepHead[start + curNum * 2 + 1], deepHead[start + nxtNum * 2], deepHead[start + nxtNum * 2 + 1]);
  if (num_tails < 1 || point_distance(deepHead[start + curNum * 2], deepHead[start + curNum * 2 + 1], deepHead[start + nxtNum * 2], deepHead[start + nxtNum * 2 + 1]) < lower)
    return;

  int centerpoint_x = x;
  int centerpoint_y = y;
  // float vein_length = vein[index(y, x) * line + curNum];
  float vein_length = vein[indexchannel(y, x, curNum)];
  float theta_cur = thetas[curNum];
  float theta_nxt = thetas[nxtNum];
  point curcoors;
  point nxtcoors;

  int flag = 0;
  for (int i = 0; i < num_tails; i++)
  {
    float alpha = 1.0 / ((num_tails - i + 1) * 2 - 1);
    float beta = 1.0 / ((num_tails - i + 1) * 2 - 2);

    // 下面更新中心点坐标
    int centerpoint_x0 = int(vein_length * alpha * cos(theta_cur) + centerpoint_x);
    int centerpoint_y0 = int(vein_length * alpha * sin(theta_cur) + centerpoint_y);
    if (check(centerpoint_x0, centerpoint_y0, H, W) == 0)
      break;

    // float vein_length_0 = vein[index(centerpoint_y0, centerpoint_x0) * line + nxtNum];
    float vein_length_0 = vein[indexchannel(centerpoint_y0, centerpoint_x0, nxtNum)];
    int x_0 = int(vein_length_0 * cos(theta_nxt) + centerpoint_x0);
    int y_0 = int(vein_length_0 * sin(theta_nxt) + centerpoint_y0);
    x_0 = clip(x_0, W - 1);
    y_0 = clip(y_0, H - 1);
    // 更新 当前位置坐标(centerpoint_x1, centerpoint_y1)，并计算该位置下所有方向与轮廓的距离
    int centerpoint_x1 = int(vein_length_0 * beta * cos(theta_nxt) + centerpoint_x0);
    int centerpoint_y1 = int(vein_length_0 * beta * sin(theta_nxt) + centerpoint_y0);
    //   计算当前位置下，指定方向下与轮廓的交点坐标

    //  增加clip----------------------------
    if (check(centerpoint_x1, centerpoint_y1, H, W) == 0)
      break;
    // # 增加clip----------------------------

    // float vein_length_1 = vein[index(centerpoint_y1, centerpoint_x1) * line + curNum];
    float vein_length_1 = vein[indexchannel(centerpoint_y1, centerpoint_x1, curNum)];
    if (vein_length_1 > upper)
      break;

    int x_1 = int(vein_length_1 * cos(theta_cur) + centerpoint_x1);
    int y_1 = int(vein_length_1 * sin(theta_cur) + centerpoint_y1);
    x_1 = clip(x_1, W - 1);
    y_1 = clip(y_1, H - 1);

    // 更新中心点 和 length
    centerpoint_x = centerpoint_x1;
    centerpoint_y = centerpoint_y1;
    vein_length = vein_length_1;
    nxtcoors.x = x_0;
    nxtcoors.y = y_0;
    curcoors.x = x_1;
    curcoors.y = y_1;

    if (point_distance(x_0, y_0, x_1, y_1) < lower || vein_length_0 < lower || vein_length_1 < lower)
      break;
  }
  // printf("\n%f %f %f %f 2\n", curcoors.x, curcoors.y, nxtcoors.x, nxtcoors.y);

  res[index(root_y, root_x) * channels + perline * root_curNum + 4 + 6 * curIndex] = curcoors.x;
  res[index(root_y, root_x) * channels + perline * root_curNum + 5 + 6 * curIndex] = curcoors.y;
  res[index(root_y, root_x) * channels + perline * root_curNum + 6 + 6 * curIndex] = nxtcoors.x;
  res[index(root_y, root_x) * channels + perline * root_curNum + 7 + 6 * curIndex] = nxtcoors.y;
}

__device__ int get_deepcoors(const float *vein, int *head, int curNum, int nxtNum, float *res, int center_x, int center_y, int H, int W,
                             int upper, int lower, const float *thetas, int line, int root_x, int root_y, int *deepHead, int *indexlist)
{
  int num_tails = 1;
  int thread = threadIdx.x;
  int count = get_indexdeeps(head[thread * line * 2 + curNum * 2],
                             head[thread * line * 2 + curNum * 2 + 1],
                             head[thread * line * 2 + nxtNum * 2],
                             head[thread * line * 2 + nxtNum * 2 + 1],
                             center_x, center_y, thetas, line, curNum, indexlist);
  if (count == 0)
    return 0;

  // int *deephead = new int[line * 2];
  int start = (thread * line + curNum) * line * 2;
  get_headcoor(vein, center_x, center_y, deepHead, thetas, H, W, line, start);
  start = threadIdx.x * line * line + curNum * line;
  for (int i = 0; i < count - 1; i++)
  {
    int curId = indexlist[start + i];
    int nxtId = indexlist[start + i + 1];
    get_deepcoors2(vein, curId, nxtId, res, center_x, center_y, H, W, num_tails, upper, lower, thetas, line, root_x, root_y, curNum, i, deepHead);
  }
  int finalIndex = indexlist[start + count - 1];
  start = (threadIdx.x * line + curNum) * line * 2;
  res[index(root_y, root_x) * channels + perline * curNum + 2 + 6 * (count - 1)] = deepHead[start + finalIndex * 2];
  res[index(root_y, root_x) * channels + perline * curNum + 3 + 6 * (count - 1)] = deepHead[start + finalIndex * 2 + 1];
  return 1;
}

__device__ void get_tailcoors(const float *vein, int curNum, int nxtNum, float *res, int x, int y, int H, int W, int num_tails,
                              int upper, int lower, const float *thetas, int line, int *head, int *deepHead, int *indexlist)
{

  //   首先将基准点放到结果里面
  int thread = threadIdx.x;
  res[index(y, x) * channels + perline * curNum] = head[thread * line * 2 + curNum * 2];
  res[index(y, x) * channels + perline * curNum + 1] = head[thread * line * 2 + curNum * 2 + 1];
  // printf("\n%d %d %d %d\n", head[thread * line * 2 + curNum * 2], head[thread * line * 2 + curNum * 2 + 1], head[thread * line * 2 + nxtNum * 2], head[thread * line * 2 + nxtNum * 2 + 1]);
  // res[indexchannel(y, x, perline * curNum)] = head[thread * line * 2 + curNum * 2];
  // res[indexchannel(y, x, perline * curNum + 1)] = head[thread * line * 2 + curNum * 2 + 1];

  if (num_tails < 1 || point_distance(head[thread * line * 2 + curNum * 2], head[thread * line * 2 + curNum * 2 + 1], head[thread * line * 2 + nxtNum * 2], head[thread * line * 2 + nxtNum * 2 + 1]) < lower)
    return;

  int centerpoint_x = x;
  int centerpoint_y = y;
  float vein_length = vein[indexchannel(y, x, curNum)];
  float theta_cur = thetas[curNum];
  float theta_nxt = thetas[nxtNum];
  point curcoors;
  point nxtcoors;

  int flag = 0;
  for (int i = 0; i < num_tails; i++)
  {
    float alpha = 1.0 / ((num_tails - i + 1) * 2 - 1);
    float beta = 1.0 / ((num_tails - i + 1) * 2 - 2);

    // 下面更新中心点坐标
    int centerpoint_x0 = int(vein_length * alpha * cos(theta_cur) + centerpoint_x);
    int centerpoint_y0 = int(vein_length * alpha * sin(theta_cur) + centerpoint_y);
    if (check(centerpoint_x0, centerpoint_y0, H, W) == 0)
      break;
    float vein_length_0 = vein[indexchannel(centerpoint_y0, centerpoint_x0, nxtNum)];
    // printf("\n%d %d %f\n", centerpoint_x0,centerpoint_y0, vein_length_0);
    int x_0 = int(vein_length_0 * cos(theta_nxt) + centerpoint_x0);
    int y_0 = int(vein_length_0 * sin(theta_nxt) + centerpoint_y0);
    // printf("\n%d %d %f %f\n", x_0, y_0, cos(theta_nxt), sin(theta_nxt));
    x_0 = clip(x_0, W - 1);
    y_0 = clip(y_0, H - 1);
    // 更新 当前位置坐标(centerpoint_x1, centerpoint_y1)，并计算该位置下所有方向与轮廓的距离
    int centerpoint_x1 = int(vein_length_0 * beta * cos(theta_nxt) + centerpoint_x0);
    int centerpoint_y1 = int(vein_length_0 * beta * sin(theta_nxt) + centerpoint_y0);
    //   计算当前位置下，指定方向下与轮廓的交点坐标

    //  增加clip----------------------------
    if (check(centerpoint_x1, centerpoint_y1, H, W) == 0)
      break;
    // # 增加clip----------------------------

    // float vein_length_1 = vein[index(centerpoint_y1, centerpoint_x1) * line + curNum];
    float vein_length_1 = vein[indexchannel(centerpoint_y1, centerpoint_x1, curNum)];
    if (vein_length_1 > upper)
      break;

    int x_1 = int(vein_length_1 * cos(theta_cur) + centerpoint_x1);
    int y_1 = int(vein_length_1 * sin(theta_cur) + centerpoint_y1);
    x_1 = clip(x_1, W - 1);
    y_1 = clip(y_1, H - 1);

    // 更新中心点 和 length
    centerpoint_x = centerpoint_x1;
    centerpoint_y = centerpoint_y1;
    vein_length = vein_length_1;
    nxtcoors.x = x_0;
    nxtcoors.y = y_0;
    curcoors.x = x_1;
    curcoors.y = y_1;

    double thetasPhiCur = point_phi(x, y, head[thread * line * 2 + curNum * 2], head[thread * line * 2 + curNum * 2 + 1]);
    double thetasPhiNxt = point_phi(x, y, head[thread * line * 2 + nxtNum * 2], head[thread * line * 2 + nxtNum * 2 + 1]);
    double thetasPhiDeep = point_phi(x, y, centerpoint_x, centerpoint_y);

    if (thetasPhiCur == 180)
      thetasPhiCur = -180;
    if (thetasPhiNxt == 180)
      thetasPhiNxt = -180;
    if (thetasPhiDeep == 180)
      thetasPhiDeep = -180;

    //！bug！当原始中心点和深度搜索的中心点角度在一定范围内，并且深度搜索的中心点不会在目标轮廓线上时，视为有效！bug！
    if (judge_deep(thetasPhiDeep, thetasPhiCur, thetasPhiNxt, vein_length_0, vein_length_1, lower) == 1)
    {
      if (i == num_tails - 1)
        flag = get_deepcoors(vein, head, curNum, nxtNum, res, centerpoint_x, centerpoint_y, H, W, upper, lower, thetas, line, x, y, deepHead, indexlist);
    }
    else
    {
      break;
    }
  }
  if (flag == 0)
  {
    res[index(y, x) * channels + perline * curNum + 2] = curcoors.x;
    res[index(y, x) * channels + perline * curNum + 3] = curcoors.y;
    res[index(y, x) * channels + perline * curNum + 4] = nxtcoors.x;
    res[index(y, x) * channels + perline * curNum + 5] = nxtcoors.y;
  }
}

__device__ void vein_gen(int x, int y, const float *vein, int upper, int lower, const float *thetas, float *res, int H, int W, int line, int *head, int *deepHead, int *indexlist)
{
  int num_tails = 1;
  int thread = threadIdx.x;
  int start = thread * line * 2;
  get_headcoor(vein, x, y, head, thetas, H, W, line, start);
  for (int i = 0; i < line; i++)
  {
    int curNum = i;
    int nxtNum = i + 1;
    if (i == line - 1)
      nxtNum = 0;
    get_tailcoors(vein, curNum, nxtNum, res, x, y, H, W, num_tails, upper, lower, thetas, line, head, deepHead, indexlist);
  }
}

__global__ void kernel(const float *centers, const float *vein, int upper, int lower, const float *thetas, float *res, int H, int W, int line, int count, int *head, int *deepHead, int *indexlist)
{

  int thread = threadIdx.x;
  if (thread >= count)
    return;
  int x = centers[thread * 2];
  int y = centers[thread * 2 + 1];
  vein_gen(x, y, vein, upper, lower, thetas, res, H, W, line, head, deepHead, indexlist);
}

void veinmask_cuda(const at::Tensor &points, const at::Tensor &distance, int upper, int lower, const at::Tensor &inf, at::Tensor &coor)
{
  AT_ASSERTM(distance.type().is_cuda(), "boxes must be a CUDA tensor");
  int counts = points.size(0);

  float *vein = distance.data_ptr<float>();
  float *centers = points.data_ptr<float>();
  float *thetas = inf.data_ptr<float>();

  int line = distance.size(2);
  int H = distance.size(0);
  int W = distance.size(1);
  float *res = coor.data<float>();

  int *head = nullptr;
  int *deepHead = nullptr;
  int *indexlist = nullptr;

  cudaMalloc((void **)&head, counts * line * 2 * sizeof(int));
  cudaMalloc((void **)&deepHead, counts * line * line * 2 * sizeof(int));
  cudaMalloc((void **)&indexlist, counts * line * line * sizeof(int));

  int threads = max(counts, 1);
  if (counts % 32 != 0)
    threads = (counts / 32 + 1) * 32;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<1, threads, 0, stream>>>(centers, vein, upper, lower, thetas, res, H, W, line, counts, head, deepHead, indexlist);
  // cudaError_t errSync = cudaGetLastError();
  // cudaError_t errAsync = cudaDeviceSynchronize();
  cudaFree(head);
  cudaFree(deepHead);
  cudaFree(indexlist);
  // if (errSync != cudaSuccess)
  //   std::cerr << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
  // if (errAsync != cudaSuccess)
  //   std::cerr << "Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
}