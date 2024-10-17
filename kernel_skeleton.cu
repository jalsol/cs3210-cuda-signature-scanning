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
constexpr int INF = MAX_SAMP_LEN * NUM_SAMPS;
constexpr int NUM_THREADS = 32;

struct View {
	int start;
	int len;
};

__device__ int min3(int a, int b, int c) {
    return min(a, min(b, c));
}

__global__ void test_match(
	const char* const __restrict__ samp_buffer,
	const View samp_view,
	const char* const __restrict__ sign_buffer,
	const View sign_view,
	const int piece_size,
	int* const answer
) {
	const int window_start = samp_view.start + blockIdx.x;
	const int window_end   = window_start + sign_view.len;

	const int piece_start = threadIdx.x * piece_size;
	const int piece_end = min3(piece_start + piece_size, sign_view.len, window_end - window_start);

	// shall we assert(piece_start < sig_size)?
	// can't go wrong unless division goes wrong

	bool match = true;

	for (int i = piece_start; i < piece_end; ++i) {
		const char samp_char = samp_buffer[window_start + i];
		const char sign_char = sign_buffer[sign_view.start + i];
		match = match && (sign_char == samp_char || sign_char == 'N' || samp_char == 'N');
	}

	// note to self: NUM_THREADS == 32
	// thus 1 block = 1 warp
    const bool all_match = __all_sync(0xFFFFFFFF, match);

	if (threadIdx.x == 0 && all_match) {
		atomicMin(answer, window_start);
	}
}

int find_match(
	const char* const __restrict__ samp_buffer,
	const View samp_view,
	const char* const __restrict__ sign_buffer,
	const View sign_view,
	int* const answer
) {
	*answer = INF;

	const int num_windows = samp_view.len - sign_view.len + 1;
	const int piece_size = (sign_view.len + NUM_THREADS - 1) / NUM_THREADS;

	test_match<<<num_windows, NUM_THREADS>>>(samp_buffer, samp_view, sign_buffer, sign_view, piece_size, answer);
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	return *answer;
}

__global__ void calc_score(const char* const __restrict__ qual_buffer, const int start, const int len, double* const output) {
	int sum = 0;

	for (int i = start; i < start + len; ++i) {
		sum += qual_buffer[i] - 33;
	}

	*output = static_cast<double>(sum) / len;
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
	char* host_samp_buffer;
	char* host_qual_buffer;
	char* host_sign_buffer;

	host_samp_buffer = reinterpret_cast<char*>(std::malloc(MAX_SAMP_LEN * NUM_SAMPS));
	host_qual_buffer = reinterpret_cast<char*>(std::malloc(MAX_SAMP_LEN * NUM_SAMPS));
	host_sign_buffer = reinterpret_cast<char*>(std::malloc(MAX_SIGN_LEN * NUM_SIGNS));

	int samp_start = 0;
	for (const auto& samp : samples) {
		const auto len = samp.seq.size();
		std::memcpy(host_samp_buffer + samp_start, samp.seq.data(),  len);
		std::memcpy(host_qual_buffer + samp_start, samp.qual.data(), len);
		samp_start += len;
	}

	int sign_start = 0;
	for (const auto& sign : signatures) {
		const auto len = sign.seq.size();
		std::memcpy(host_sign_buffer + sign_start, sign.seq.data(), len);
		sign_start += len;
	}

	char* samp_buffer;
	char* qual_buffer;
	char* sign_buffer;

	CHECK_CUDA_ERROR(cudaMalloc(&samp_buffer, samp_start));
	CHECK_CUDA_ERROR(cudaMalloc(&qual_buffer, samp_start));
	CHECK_CUDA_ERROR(cudaMalloc(&sign_buffer, sign_start));

	CHECK_CUDA_ERROR(cudaMemcpy(samp_buffer, host_samp_buffer, samp_start, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(qual_buffer, host_qual_buffer, samp_start, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(sign_buffer, host_sign_buffer, sign_start, cudaMemcpyHostToDevice));

	std::free(host_samp_buffer);
	std::free(host_qual_buffer);
	std::free(host_sign_buffer);

	double* match_score;
	int* match_pos;
	CHECK_CUDA_ERROR(cudaMallocManaged(&match_score, sizeof(double)));
	CHECK_CUDA_ERROR(cudaMallocManaged(&match_pos,   sizeof(int)));

	samp_start = 0;
	for (const auto& samp : samples) {
		const View samp_view = { samp_start, static_cast<int>(samp.seq.size()) };
		sign_start = 0;

		for (const auto& sign : signatures) {
			const View sign_view = { sign_start, static_cast<int>(sign.seq.size()) };
			find_match(samp_buffer, samp_view, sign_buffer, sign_view, match_pos);

			if (*match_pos != INF) {
				calc_score<<<1, 1>>>(qual_buffer, *match_pos, sign_view.len, match_score);
				CHECK_CUDA_ERROR(cudaDeviceSynchronize());
				matches.push_back({ samp.name, sign.name, *match_score });
			}

			sign_start += sign_view.len;
		}

		samp_start += samp_view.len;
	}

	CHECK_CUDA_ERROR(cudaFree(match_score));
	CHECK_CUDA_ERROR(cudaFree(match_pos));

	CHECK_CUDA_ERROR(cudaFree(samp_buffer));
	CHECK_CUDA_ERROR(cudaFree(qual_buffer));
	CHECK_CUDA_ERROR(cudaFree(sign_buffer));
}
