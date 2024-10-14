#include "kseq/kseq.h"
#include "common.h"

#include <string_view>

int find_match(std::string_view sample, std::string_view signature) {
	const int sig_size = std::ssize(signature);
	const int samp_size = std::ssize(sample);

	for (int i = 0; i + sig_size < samp_size; ++i) {
		int offset = 0;

		for (; offset < sig_size; ++offset) {
			if (signature[offset] == 'N' || sample[i + offset] == 'N') {
				continue;
			}

			if (signature[offset] != sample[i + offset]) {
				break;
			}
		}

		if (offset >= sig_size) {
			return i;
		}
	}

	return -1;
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
		for (const auto& sig : signatures) {
			const auto match_pos = find_match(samp.seq, sig.seq);
			if (match_pos != -1) {
				const auto match_score = calc_score(samp.qual, match_pos, std::ssize(sig.seq));
				matches.push_back({ samp.name, sig.name, match_score });
			}
		}
	}
}
