#include "kseq/kseq.h"
#include "common.h"

#include <string_view>
#include <iostream>

#ifdef ENABLE_CUDA_CHECK
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#else
#define CHECK_CUDA_ERROR(val) (val)
#endif // ENABLE_CUDA_CHECK

// ===================================================================================

constexpr int MAX_SAMP_LEN = 200'000;
constexpr int MAX_SIGN_LEN = 10'000;
constexpr int NUM_SAMPS = 2200;
constexpr int NUM_SIGNS = 1000;
// constexpr int INF = MAX_SAMP_LEN * NUM_SAMPS;
// constexpr int NUM_THREADS = 32;

struct View {
	int start;
	int len;
};

__device__ void calc_score(const char* const __restrict__ qual_buffer, const int start, const int len, double* const __restrict__ output);
__global__ void init_array(double* const __restrict__ array, const int size);

__global__ void find_match(
	const char* const __restrict__ samp_buffer,
	const char* const __restrict__ sign_buffer,
	const char* const __restrict__ qual_buffer,
	const int*  const __restrict__ pref_samp,
	const int*  const __restrict__ pref_sign,
	double*     const __restrict__ score_buffer
) {
	const int samp_idx = blockIdx.x;
	const int sign_idx = blockIdx.y;
	const int pair_idx = samp_idx * gridDim.y + sign_idx;

	const int samp_start = pref_samp[samp_idx];
	const int samp_next  = pref_samp[samp_idx + 1];
	const int sign_start = pref_sign[sign_idx];
	const int sign_next  = pref_sign[sign_idx + 1];

	const View samp_view { samp_start, samp_next - samp_start };
	const View sign_view { sign_start, sign_next - sign_start };

	for (int offset = 0; offset + sign_view.len < samp_view.len; ++offset) {
		int ptr = 0;

		for (; ptr < sign_view.len; ++ptr) {
			const char samp_char = samp_buffer[samp_view.start + ptr + offset];
			const char sign_char = sign_buffer[sign_view.start + ptr];

			if (samp_char != sign_char && samp_char != 'N' && sign_char != 'N') {
				break;
			}
		}

		if (ptr >= sign_view.len) {
			calc_score(qual_buffer, samp_view.start + offset, sign_view.len, score_buffer + pair_idx);
			break;
		}
	}
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
	char* host_sign_buffer;
	int*  host_pref_sign;

	CHECK_CUDA_ERROR(cudaMallocHost(&host_sign_buffer, MAX_SIGN_LEN * NUM_SIGNS));
	CHECK_CUDA_ERROR(cudaMallocHost(&host_pref_sign,    sizeof(int) * (signatures.size() + 1)));

	char* samp_buffer;
	char* qual_buffer;
	int*  pref_samp;

	CHECK_CUDA_ERROR(cudaMallocManaged(&samp_buffer, MAX_SAMP_LEN * NUM_SAMPS));
	CHECK_CUDA_ERROR(cudaMallocManaged(&qual_buffer, MAX_SAMP_LEN * NUM_SAMPS));
	CHECK_CUDA_ERROR(cudaMallocManaged(&pref_samp,   sizeof(int)  * (samples.size() + 1)));

	pref_samp[0] = 0;
	host_pref_sign[0] = 0;

	for (int i = 1; i <= samples.size(); ++i) {
		const auto& samp = samples[i - 1];
		const auto len   = samp.seq.size();
		const auto prev  = pref_samp[i - 1];

		cudaMemcpy(samp_buffer + prev, samp.seq.data(),  len, cudaMemcpyDefault);
		cudaMemcpy(qual_buffer + prev, samp.qual.data(), len, cudaMemcpyDefault);
		pref_samp[i] = prev + len;
	}

	for (int i = 1; i <= signatures.size(); ++i) {
		const auto& sign = signatures[i - 1];
		const auto len   = sign.seq.size();

		cudaMemcpy(host_sign_buffer + host_pref_sign[i - 1], sign.seq.data(), len, cudaMemcpyDefault);
		host_pref_sign[i] = host_pref_sign[i - 1] + len;
	}

	char*   sign_buffer;
	int*    pref_sign;
	double* score_buffer;

	const auto num_pairs      = samples.size() * signatures.size();
	const auto total_sign_len = host_pref_sign[signatures.size()];

	CHECK_CUDA_ERROR(cudaMalloc(&sign_buffer,  total_sign_len));
	CHECK_CUDA_ERROR(cudaMalloc(&pref_sign,    sizeof(int) * (signatures.size() + 1)));
	CHECK_CUDA_ERROR(cudaMalloc(&score_buffer, sizeof(double) * num_pairs));

	CHECK_CUDA_ERROR(cudaMemcpy(sign_buffer, host_sign_buffer, total_sign_len,                        cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(pref_sign,   host_pref_sign,   sizeof(int) * (signatures.size() + 1), cudaMemcpyHostToDevice));
	init_array<<<(num_pairs + 255) / 256, 256>>>(score_buffer, num_pairs);

	CHECK_CUDA_ERROR(cudaFreeHost(host_sign_buffer));
	CHECK_CUDA_ERROR(cudaFreeHost(host_pref_sign));

	const dim3 grid_size(samples.size(), signatures.size());
	find_match<<<grid_size, 1>>>(
		samp_buffer,
		sign_buffer,
		qual_buffer,
		pref_samp,
		pref_sign,
		score_buffer
	);

	double* result;
	CHECK_CUDA_ERROR(cudaMallocHost(&result, sizeof(double) * num_pairs));
	CHECK_CUDA_ERROR(cudaMemcpy(result, score_buffer, sizeof(double) * num_pairs, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(score_buffer));

	for (int i = 0; i < samples.size(); ++i) {
		for (int j = 0; j < signatures.size(); ++j) {
			const int pair_idx = i * signatures.size() + j;

			if (result[pair_idx] != -1.0) {
				matches.push_back({ samples[i].name, signatures[j].name, result[pair_idx] });
			}
		}
	}

	CHECK_CUDA_ERROR(cudaFreeHost(result));

	CHECK_CUDA_ERROR(cudaFree(samp_buffer));
	CHECK_CUDA_ERROR(cudaFree(qual_buffer));
	CHECK_CUDA_ERROR(cudaFree(sign_buffer));
	CHECK_CUDA_ERROR(cudaFree(pref_samp));
	CHECK_CUDA_ERROR(cudaFree(pref_sign));
}

__device__ void calc_score(
	const char* const __restrict__ qual_buffer,
	const int start,
	const int len,
	double* const __restrict__ output
) {
	int sum = 0;

	for (int i = start; i < start + len; ++i) {
		sum += qual_buffer[i] - 33;
	}

	*output = static_cast<double>(sum) / len;
}

__global__ void init_array(double* const __restrict__ array, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = -1.0;
    }
}
