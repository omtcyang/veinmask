#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

// template <typename T>
// class cudaQueue
// {
// public:
// 	T *main_Mem = nullptr;
// 	T *temp_Mem = nullptr;
// 	int SIZE = 0;
// 	CUDA_CALLABLE_MEMBER cudaQueue() = default;

// 	CUDA_CALLABLE_MEMBER cudaQueue(cudaQueue &&lQueue)
// 	{
// 		this->SIZE = lQueue.SIZE;
// 		this->main_Mem = lQueue.main_Mem;
// 		lQueue.main_Mem = nullptr;
// 		lQueue.SIZE = 0;
// 	}

// 	CUDA_CALLABLE_MEMBER cudaQueue(cudaQueue &lQueue)
// 	{
// 		this->SIZE = lQueue.size();
// 		for (int idx = 0; idx < this->SIZE; ++idx)
// 			this->main_Mem[idx] = lQueue.main_Mem[idx];
// 	}

// 	CUDA_CALLABLE_MEMBER inline void push(T value)
// 	{
// 		if (SIZE == 0)
// 		{
// 			main_Mem = new T[++SIZE];
// 			main_Mem[0] = value;
// 		}
// 		else
// 		{
// 			temp_Mem = new T[SIZE];
// 			memcpy((void *)temp_Mem, (void *)main_Mem, sizeof(T) * SIZE);
// 			delete[] main_Mem;
// 			main_Mem = new T[SIZE + 1];
// 			main_Mem[SIZE] = value;
// 			memcpy((void *)main_Mem, (void *)temp_Mem, sizeof(T) * SIZE);
// 			SIZE++;
// 			delete[] temp_Mem;
// 		}
// 	}

// 	CUDA_CALLABLE_MEMBER inline void insert_0(T value)
// 	{
// 		if (SIZE == 0)
// 		{
// 			main_Mem = new T[++SIZE];
// 			main_Mem[0] = value;
// 		}
// 		else
// 		{
// 			temp_Mem = new T[SIZE];
// 			memcpy((void *)temp_Mem, (void *)main_Mem, sizeof(T) * SIZE);
// 			delete[] main_Mem;
// 			main_Mem = new T[++SIZE];
// 			*main_Mem = value;
// 			memcpy((void *)(main_Mem + 1), (void *)temp_Mem, sizeof(T) * SIZE);
// 			delete[] temp_Mem;
// 		}
// 	}

// 	CUDA_CALLABLE_MEMBER inline void pop()
// 	{
// 		if (SIZE == 0)
// 		{
// 			printf("\nPopping empty Queue\n");
// 			exit(-1); // popping empty Queue
// 		}
// 		temp_Mem = new T[SIZE];
// 		memcpy((void *)temp_Mem, (void *)main_Mem, sizeof(T) * SIZE);
// 		delete[] main_Mem;
// 		main_Mem = new T[--SIZE];
// 		memcpy((void *)main_Mem, (void *)(temp_Mem + 1), sizeof(T) * SIZE);
// 		delete[] temp_Mem;
// 	}

// 	CUDA_CALLABLE_MEMBER inline T front()
// 	{
// 		if (SIZE == 0)
// 		{
// 			printf("\nNo element in Queue\n");
// 			exit(-1); // no element in Queue
// 		}
// 		return main_Mem[0];
// 	}

// 	CUDA_CALLABLE_MEMBER inline T get(int index)
// 	{
// 		if (SIZE <= index)
// 		{
// 			printf("\nerror\n");
// 			exit(-1);
// 		}

// 		return main_Mem[index];
// 	}

// 	CUDA_CALLABLE_MEMBER inline bool empty()
// 	{
// 		return SIZE == 0;
// 	}
// 	CUDA_CALLABLE_MEMBER inline int size()
// 	{
// 		return SIZE;
// 	}

// 	CUDA_CALLABLE_MEMBER ~cudaQueue()
// 	{
// 		delete[] main_Mem;
// 	}
// };

struct point
{
	/* data */
	int x, y;
	CUDA_CALLABLE_MEMBER point() {}

	CUDA_CALLABLE_MEMBER point(int a, int b)
	{
		x = a;
		y = b;
	}
};
