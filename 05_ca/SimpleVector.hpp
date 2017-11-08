//   Copyright 2017, Felice Pantaleo, CERN
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef GPU_SIMPLEVECTOR_HPP_
#define GPU_SIMPLEVECTOR_HPP_

namespace GPU
{
	template<class T>
	struct SimpleVector
	{
    // Constructors
    __host__ __device__
    SimpleVector(unsigned int maxSize, T *m_data = nullptr)
        : m_size(0), m_data(m_data), maxSize(static_cast<int>(maxSize)) {}

    __host__ __device__
    SimpleVector() : SimpleVector(0) {}

		__inline__
		  __host__ __device__
		int push_back(const T& element)
		{

			auto previousSize = m_size;
			m_size++;
			if(previousSize < maxSize)
			{
				m_data[previousSize] = element;
				return previousSize;
			}
			else
			{
				--m_size;
				return -1;
			}
		}

#if defined(__NVCC__) || defined(__CUDACC__)
		__device__
		int push_back_ts(const T& element)
		{
			auto previousSize = atomicAdd(&m_size, 1);
			if(previousSize<maxSize)
			{
				m_data[previousSize] = element;
				return previousSize;
			}
			else
			{
				atomicSub(&m_size, 1);
				return -1;
			}
		}
#endif

		__inline__
		  __host__ __device__
		T pop_back()
		{
			if(m_size > 0)
			{
				auto previousSize = m_size--;
				return m_data[previousSize-1];
			}
			else
			return T();
		}

		__inline__   __host__   __device__
		void reset()
		{
			m_size = 0;
		}


		__inline__   __host__   __device__
		int size() const
		{
			return m_size;
		}

		int m_size;

		T *m_data;

    int maxSize;

	};
}

#endif
