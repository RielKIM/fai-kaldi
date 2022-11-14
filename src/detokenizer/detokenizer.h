#include <string>
#include <sstream>
#include <vector>

#include "util/text-utils.h"

namespace kaldi {

	class Detokenizer{

		public:
			Detokenizer();
			bool isFront(std::string word);
			bool isEnd(std::string word);
			bool isFrontEnd(std::string word);
			std::vector<std::string> detokenizedWords(std::vector<std::string> words);
			
			~Detokenizer();
	};
}
