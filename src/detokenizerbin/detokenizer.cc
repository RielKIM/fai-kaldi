//  restore tokenized recognition words
//
//
//
//
//
//

#include "util/text-utils.h"
#include "base/kaldi-common.h"
#include "util/kaldi-table.h"
#include "util/parse-options.h"

#include "detokenizer/detokenizer.h"


int main(int argc, char *argv[]) {
	using namespace kaldi;
	std::vector<std::pair<std::string, std::string> > pair;
	std::vector<std::pair<std::string, std::string> > pair_out;
	std::pair<std::string, std::string> pair_temp;

	std::vector<std::string> words;
	std::vector<std::string> output;
	const char delim = ' ';
	Detokenizer tkn;
	std::stringstream ss_temp;

	const char * usage = "<usage> detokenizer <tokenized file> <output file> \n";
	ParseOptions po(usage);

	po.Read(argc, argv);

	if (po.NumArgs() != 2) 
	{
		po.PrintUsage();
		return 1;
	}

	std::string tok_rspecifier = po.GetArg(1);
	std::string output_wx = po.GetArg(2);

	ReadScriptFile(tok_rspecifier,false,&pair);


	for(int j=0;j<pair.size();j++)
	{
	
		SplitStringToVector(pair[j].second, &delim, true, &words);
		output = tkn.detokenizedWords(words);
		ss_temp.str("");
		pair_temp.second = "";

		for(int i=0;i<output.size();i++)
		{
			if(i < (output.size()-1))
			{
				pair_temp.second += output[i] + " ";
			}else{
				ss_temp << output[i];
				pair_temp.second += output[i];
			}
			pair_temp.first = pair[j].first;
		}
		pair_out.push_back(pair_temp);
	}

	WriteScriptFile(output_wx, pair_out);
	return -1;
}

