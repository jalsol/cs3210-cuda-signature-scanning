#include "kseq/kseq.h"
#include "common.h"

#include <string_view>

constexpr int MAX_SAMP_LEN = 200'000;
constexpr int MAX_SIGN_LEN = 10'000;
constexpr int NUM_SAMPS = 2200;
constexpr int NUM_SIGNS = 1000;
constexpr int INF = MAX_SAMP_LEN * NUM_SAMPS;

struct View {
	int start;
	int len;
};

int find_match(
	const char* const __restrict__ samp_buffer,
	const View samp_view,
	const char* const __restrict__ sign_buffer,
	const View sign_view
) {
	for (int offset = 0; offset + sign_view.len < samp_view.len; ++offset) {
		const int window_start = samp_view.start + offset; // probably enough registers for another variable...
		int ptr = 0;

		for (; ptr < sign_view.len; ++ptr) {
			char samp_char = samp_buffer[window_start + ptr];
			char sign_char = sign_buffer[sign_view.start + ptr];

			if (samp_char != sign_char && samp_char != 'N' && sign_char != 'N') {
				break;
			}
		}

		if (ptr >= sign_view.len) {
			return window_start;
		}
	}

	return INF;
}

void calc_score(const char* const __restrict__ qual_buffer, const int start, const int len, double* output) {
	int sum = 0;

	for (int i = start; i < start + len; ++i) {
		sum += qual_buffer[i] - 33;
	}

	*output = static_cast<double>(sum) / len;
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
	char* samp_buffer;
	char* qual_buffer;
	char* sign_buffer;

	samp_buffer = reinterpret_cast<char*>(std::malloc(MAX_SAMP_LEN * NUM_SAMPS));
	qual_buffer = reinterpret_cast<char*>(std::malloc(MAX_SAMP_LEN * NUM_SAMPS));
	sign_buffer = reinterpret_cast<char*>(std::malloc(MAX_SIGN_LEN * NUM_SIGNS));

	{
		int samp_start = 0;
		for (const auto& samp : samples) {
			const auto len = samp.seq.size();
			std::memcpy(samp_buffer + samp_start, samp.seq.data(),  len);
			std::memcpy(qual_buffer + samp_start, samp.qual.data(), len);
			samp_start += len;
		}
	}

	{
		int sign_start = 0;
		for (const auto& sign : signatures) {
			const auto len = sign.seq.size();
			std::memcpy(sign_buffer + sign_start, sign.seq.data(), len);
			sign_start += len;
		}
	}

	int samp_start = 0;
	for (const auto& samp : samples) {
		const View samp_view = { samp_start, static_cast<int>(samp.seq.size()) };
		int sign_start = 0;

		for (const auto& sign : signatures) {
			const View sign_view = { sign_start, static_cast<int>(sign.seq.size()) };
			const auto match_pos = find_match(samp_buffer, samp_view, sign_buffer, sign_view);

			if (match_pos != INF) {
				double match_score;
				calc_score(qual_buffer, match_pos, sign_view.len, &match_score);
				matches.push_back({ samp.name, sign.name, match_score });
			}

			sign_start += sign_view.len;
		}

		samp_start += samp_view.len;
	}

	std::free(samp_buffer);
	std::free(qual_buffer);
	std::free(sign_buffer);
}
