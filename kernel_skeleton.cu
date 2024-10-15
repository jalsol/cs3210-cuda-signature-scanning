#include "kseq/kseq.h"
#include "common.h"

#include <string_view>

constexpr int MAX_BUFFER = 200000;

char sig_buffer[MAX_BUFFER];
char samp_buffer[MAX_BUFFER];
int device_answer;

void test_block(const int tid, const int samp_size, const int sig_size, const int offset) {
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
		device_answer = std::min(device_answer, offset + start);
	}
}

int find_match(const int samp_size, std::string_view signature) {
	const int sig_size = std::ssize(signature);
	int answer = MAX_BUFFER;
	std::memcpy(&device_answer, &answer, sizeof(answer));

	for (int i = 0; i < samp_size / sig_size; ++i) {
		const int start = i * sig_size;
		std::memcpy(sig_buffer + start, signature.data(), sig_size);
	}

	for (int offset = 0; offset < sig_size; ++offset) {
		const int num_threads = (samp_size - offset) / sig_size;

		for (int tid = 0; tid < num_threads; ++tid) {
			test_block(tid, samp_size, sig_size, offset);
		}
	}

	std::memcpy(&answer, &device_answer, sizeof(device_answer));

	return answer;
}

double calc_score(std::string_view quality, int start, int size) {
	int sum = 0;

	for (int i = start; i < start + size; ++i) {
		sum += quality[i] - 33;
	}

	return static_cast<double>(sum) / size;
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
	for (const auto& samp : samples) {
		const int samp_size = std::ssize(samp.seq);
		std::memcpy(samp_buffer, samp.seq.data(), samp_size);

		for (const auto& sig : signatures) {
			const auto match_pos = find_match(samp_size, sig.seq);

			if (match_pos != MAX_BUFFER) {
				const auto match_score = calc_score(samp.qual, match_pos, std::ssize(sig.seq));
				matches.push_back({ samp.name, sig.name, match_score });
			}
		}
	}
}
