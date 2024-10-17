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

constexpr int MAX_SAMPLE_LEN = 200000;
constexpr int MAX_SIGNATURE_LEN = 10000;
constexpr int NUM_THREADS = 32; // 1 block = 1 warp

char* sig_buffer;
char* samp_buffer;
__device__ int device_answer;

__device__ int min3(int a, int b, int c) {
    return min(a, min(b, c));
}

__global__ void test_match(
	const char* const __restrict__ samp_buffer,
	const int samp_size,
	const char* const __restrict__ sig_buffer,
	const int sig_size,
	const int piece_size
) {
	const int offset = blockIdx.x; // for now blockDim.x == 1?

	const int piece_start = threadIdx.x * piece_size;
	const int piece_end = min3(piece_start + piece_size, sig_size, samp_size - offset);

	// shall we assert(piece_start < sig_size)?
	// can't go wrong unless division goes wrong

	bool match = true;

	for (int i = piece_start; i < piece_end; ++i) {
		const char samp_char = samp_buffer[offset + i];
		const char sig_char = sig_buffer[i];
		match = match && (sig_char == samp_char || sig_char == 'N' || samp_char == 'N');
	}

	// note to self: NUM_THREADS == 32
	// thus 1 block = 1 warp
    const bool all_match = __all_sync(0xFFFFFFFF, match);

	if (threadIdx.x == 0 && all_match) {
		atomicMin(&device_answer, offset);
	}
}

int find_match(const int samp_size, const int sig_size) {
	int answer = MAX_SAMPLE_LEN;
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(device_answer, &answer, sizeof(answer)));

	const int num_windows = samp_size - sig_size + 1;
	const int piece_size = (sig_size + NUM_THREADS - 1) / NUM_THREADS;

	test_match<<<num_windows, NUM_THREADS>>>(samp_buffer, samp_size, sig_buffer, sig_size, piece_size);

	CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&answer, device_answer, sizeof(answer)));
	return answer;
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
	CHECK_CUDA_ERROR(cudaMalloc(&samp_buffer, MAX_SAMPLE_LEN));
	CHECK_CUDA_ERROR(cudaMalloc(&sig_buffer, MAX_SIGNATURE_LEN));

	for (const auto& samp : samples) {
		const int samp_size = std::ssize(samp.seq);
		CHECK_CUDA_ERROR(cudaMemcpy(samp_buffer, samp.seq.data(), samp_size, cudaMemcpyHostToDevice));

		for (const auto& sig : signatures) {
			const int sig_size = std::ssize(sig.seq);
			CHECK_CUDA_ERROR(cudaMemcpy(sig_buffer, sig.seq.data(), sig_size, cudaMemcpyHostToDevice));

			const auto match_pos = find_match(samp_size, sig_size);
			if (match_pos != MAX_SAMPLE_LEN) {
				const auto match_score = calc_score(samp.qual, match_pos, std::ssize(sig.seq));
				matches.push_back({ samp.name, sig.name, match_score });
			}
		}
	}

	CHECK_CUDA_ERROR(cudaFree(samp_buffer));
	CHECK_CUDA_ERROR(cudaFree(sig_buffer));
}
