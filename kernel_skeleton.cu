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

double calc_score(std::string_view quality, int start, int size) {
	int sum = 0;

	for (int i = start; i < start + size; ++i) {
		sum += quality[i] - 33;
	}

	return static_cast<double>(sum) / size;
}


// ===================================================================================

constexpr int MAX_BUFFER = 200000;

char* sig_buffer;
char* samp_buffer;
__device__ int device_answer;

__global__ void test_block(
	const char* const __restrict__ samp_buffer,
	const int samp_size,
	const char* const __restrict__ sig_buffer,
	const int sig_size,
	const int offset
) {
	const int tid = threadIdx.x;
	const int start = tid * sig_size;
	const int end = start + sig_size;

	if (offset + end >= samp_size)
		return;

	int i;
	for (i = start; i < end; ++i) {
		if (sig_buffer[i] == 'N' || samp_buffer[i + offset] == 'N')
			continue;
		if (sig_buffer[i] != samp_buffer[i + offset])
			break;
	}

	if (i >= end) {
		atomicMin(&device_answer, offset + start);
	}
}

int find_match(const int samp_size, std::string_view signature) {
	const int sig_size = std::ssize(signature);
	int answer = MAX_BUFFER;
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(device_answer, &answer, sizeof(answer)));

	for (int i = 0; i < samp_size / sig_size; ++i) {
		const int start = i * sig_size;
		CHECK_CUDA_ERROR(cudaMemcpy(sig_buffer + start, signature.data(), sig_size, cudaMemcpyHostToDevice));
	}

	for (int offset = 0; offset < sig_size; ++offset) {
		const int num_threads = (samp_size - offset) / sig_size;
		test_block<<<1, num_threads>>>(samp_buffer, samp_size, sig_buffer, sig_size, offset);
	}

	CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&answer, device_answer, sizeof(answer)));

	return answer;
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
	CHECK_CUDA_ERROR(cudaMalloc(&samp_buffer, MAX_BUFFER));
	CHECK_CUDA_ERROR(cudaMalloc(&sig_buffer, MAX_BUFFER));

	for (const auto& samp : samples) {
		const int samp_size = std::ssize(samp.seq);
		CHECK_CUDA_ERROR(cudaMemcpy(samp_buffer, samp.seq.data(), samp_size, cudaMemcpyHostToDevice));

		for (const auto& sig : signatures) {
			const auto match_pos = find_match(samp_size, sig.seq);

			if (match_pos != MAX_BUFFER) {
				const auto match_score = calc_score(samp.qual, match_pos, std::ssize(sig.seq));
				matches.push_back({ samp.name, sig.name, match_score });
			}
		}
	}

	CHECK_CUDA_ERROR(cudaFree(samp_buffer));
	CHECK_CUDA_ERROR(cudaFree(sig_buffer));
}

