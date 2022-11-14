//  restore tokenized recognition words
//
//

#include "util/text-utils.h"
#include "base/kaldi-common.h"
#include "detokenizer/detokenizer.h"

namespace kaldi {


bool Detokenizer::isFront(std::string word)
{
	if(*word.begin() == '_')
	{
		return true;
	}else{
		return false;
	}

}

bool Detokenizer::isEnd(std::string word)
{
	if(*(word.end()-1) == '_')
	{
		return true;
	}else{
		return false;
	}
}

bool Detokenizer::isFrontEnd(std::string word)
{
	if(isFront(word) && isEnd(word))
	{
		return true;
	}else{
		return false;
	}
}

std::vector<std::string> Detokenizer::detokenizedWords(std::vector<std::string> words)
{
	std::stringstream ss_temp;
	std::vector<std::string> detokenized_vec;
	std::string temp_str;

	for(int i=0;i<words.size();i++)
	{
		if(isFrontEnd(words[i])) // case 1 "_()_"
		{
			if(ss_temp.str() != "")
			{
				detokenized_vec.push_back(ss_temp.str());
				ss_temp.str("");
			}
			ss_temp << words[i].substr(1,words[i].length()-2);

		}else if(isFront(words[i])){ // case 2 "_() "
			if(ss_temp.str() != "")
			{
				detokenized_vec.push_back(ss_temp.str());
				ss_temp.str("");
			}
			ss_temp << words[i].substr(1,words[i].length());
		}else if(isEnd(words[i])){ // case 3 " ()_" 
			ss_temp << words[i].substr(0,words[i].length()-1);
		}else{ // others

			ss_temp << words[i];
		}
	}

	if(ss_temp.str() != "")
	{
		detokenized_vec.push_back(ss_temp.str());
		ss_temp.str("");
	}

	return detokenized_vec;
}

Detokenizer::Detokenizer()
{
}

Detokenizer::~Detokenizer()
{
} 

}

