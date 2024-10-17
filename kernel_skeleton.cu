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

__device__ int device_answer;

__device__ int min3(int a, int b, int c) {
    return min(a, min(b, c));
}

__global__ void calc_score(
	const char* const __restrict__ qual_buffer,
	const int len,
	double* const __restrict__ output
) {
	const int start = device_answer;
	if (start == INF) return;

	int sum = 0;

	for (int i = start; i < start + len; ++i) {
		sum += qual_buffer[i] - 33;
	}

	*output = static_cast<double>(sum) / len;
}

__global__ void test_match(
	const char* const __restrict__ samp_buffer,
	const View samp_view,
	const char* const __restrict__ sign_buffer,
	const View sign_view,
	const char* const __restrict__ qual_buffer,
	double* const __restrict__ score_buffer,
	const int pair_idx,
	const int piece_size
) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		device_answer = INF;
	}

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
		atomicMin(&device_answer, window_start);
	}
}

void find_match(
	const char* const __restrict__ samp_buffer,
	const View samp_view,
	const char* const __restrict__ sign_buffer,
	const View sign_view,
	const char* const __restrict__ qual_buffer,
	double* const __restrict__ score_buffer,
	const int pair_idx
) {
	const int num_windows = samp_view.len - sign_view.len + 1;
	const int piece_size = (sign_view.len + NUM_THREADS - 1) / NUM_THREADS;

	test_match<<<num_windows, NUM_THREADS>>>(
		samp_buffer,
		samp_view,
		sign_buffer,
		sign_view,
		qual_buffer,
		score_buffer,
		pair_idx,
		piece_size
	);

	calc_score<<<1, 1>>>(qual_buffer, sign_view.len, score_buffer + pair_idx);
}

__global__ void init_array(double* const __restrict__ array, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = -1.0;
    }
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

	const auto num_pairs = samples.size() * signatures.size();
	double* score_buffer;
	CHECK_CUDA_ERROR(cudaMalloc(&score_buffer, sizeof(double) * num_pairs));

	init_array<<<(num_pairs + 255) / 256, 256>>>(score_buffer, num_pairs);

	for (int i = 0, samp_start = 0; i < samples.size(); ++i) {
		const auto& samp = samples[i];
		const View samp_view = { samp_start, static_cast<int>(samp.seq.size()) };

		for (int j = 0, sign_start = 0; j < signatures.size(); ++j) {
			const auto& sign = signatures[j];
			const View sign_view = { sign_start, static_cast<int>(sign.seq.size()) };
			const int pair_idx = i * signatures.size() + j;

			find_match(samp_buffer, samp_view, sign_buffer, sign_view, qual_buffer, score_buffer, pair_idx);

			sign_start += sign_view.len;
		}

		samp_start += samp_view.len;
	}

	auto* result = reinterpret_cast<double*>(std::malloc(sizeof(double) * num_pairs));
	CHECK_CUDA_ERROR(cudaMemcpy(result, score_buffer, sizeof(double) * num_pairs, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(score_buffer));

	for (int i = 0, samp_start = 0; i < samples.size(); ++i) {
		const auto& samp = samples[i];
		const View samp_view = { samp_start, static_cast<int>(samp.seq.size()) };

		for (int j = 0, sign_start = 0; j < signatures.size(); ++j) {
			const auto& sign = signatures[j];
			const View sign_view = { sign_start, static_cast<int>(sign.seq.size()) };
			const int pair_idx = i * signatures.size() + j;

			if (result[pair_idx] != -1.0) {
				matches.push_back({ samp.name, sign.name, result[pair_idx] });
			}

			sign_start += sign_view.len;
		}

		samp_start += samp_view.len;
	}


	CHECK_CUDA_ERROR(cudaFree(samp_buffer));
	CHECK_CUDA_ERROR(cudaFree(qual_buffer));
	CHECK_CUDA_ERROR(cudaFree(sign_buffer));
}
