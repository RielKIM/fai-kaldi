//
// Created by uk on 2022/11/10.
//

#ifndef KALDI_FAI_DECODER_WRAPPER_H
#define KALDI_FAI_DECODER_WRAPPER_H

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include "fai-decoder.h"

#include <typeinfo>
#include <iostream>
#include <ctime>

using namespace kaldi;
using namespace fst;

typedef kaldi::int32 int32;
typedef kaldi::int64 int64;

class FaiDecoderWrapper {
public:
    ~FaiDecoderWrapper();

    struct membuf : std::streambuf {
        membuf(char *begin, char *end) {
            this->setg(begin, begin, end);
        }
    };

    void decode_finalize(char *transactionid);

    void decode_finalize_manual(char *transactionid);

    void run_decode_stream(char *buffer, int buffer_size, char *transactionid);

    void run_decode_file(char *buffer, int buffer_size, char *transactionid);

    void run_decode_websocket(char *buffer, int buffer_size, char *transactionid);

    void run_decode_websocket_end(char *buffer, int buffer_size, char *transactionid);

    char *get_result();

    // faidecoder 객체를 inference 수행하기 전 상태로 reset 한다.
    void reset_onlinedecoder();

    char *get_fst_addr_string();

    char *get_word_syms_addr_string();

    char *get_feature_info_addr_string();

    char *get_am_nnet_addr_string();

    fst::Fst<fst::StdArc> *get_fst_addr();

    fst::SymbolTable *get_word_syms_addr();

    OnlineNnet2FeaturePipelineInfo *get_feature_info_addr();

    nnet3::AmNnetSimple *get_am_nnet_addr();

    float get_am_score();

    double get_lm_score();

    void get_fst_addr_to_string();

    void get_word_syms_addr_to_string();

    void get_feature_info_addr_to_string();

    void get_am_nnet_addr_to_string();

    fst::Fst<fst::StdArc> *get_fst_addr_to_pointer(char *addr_str);

    fst::SymbolTable *get_word_syms_addr_to_pointer(char *addr_str);

    OnlineNnet2FeaturePipelineInfo *get_feature_info_addr_to_pointer(char *addr_str);

    nnet3::AmNnetSimple *get_am_nnet_addr_to_pointer(char *addr_str);

    // faidecoder 객체를 생성 한다.
    void load_onlinedecoder();

    void load_onlinedecoder_addr(char *addr_fst, char *addr_word_syms, char *addr_feature_info, char *addr_am_nnet);

    //동작에 필요한 config 값들을 초기화 한다.
    //해당 값들을 confiure.ini등의 파일로 설정 할 수 있도록 해야한다.
    void init_configure(int _app_chunksize,
                        float _chunk_length_secs, bool _do_endpointing, bool _detokenized_flag, bool _confBR_flag,
                        int _nbest, float _lm_scale, float _wip, char *_silence_phones);

private:
    FaiDecoder *od;
    std::string word_syms_rxfilename;
    std::string align_wxfilename;
    std::string word_boundary_rxfilename;
    std::string phone_syms_rxfilename;

    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;
    WordBoundaryInfoNewOpts wb_info_opts;

    BaseFloat chunk_length_secs;
    bool do_endpointing;
    bool online;
    bool detokenized_flag;
    bool confBR_flag;
    int32 deb_lv;
    int32 nbest;
    BaseFloat lm_scale;
    BaseFloat wip;

    std::vector<std::pair<std::string, std::string>> pair_out;
    std::vector<std::pair<std::string, std::string>> pair_word_align_out;
    std::vector<std::pair<std::string, std::string>> pair_phone_align_out;

    std::string nnet3_rxfilename;
    std::string fst_rxfilename;
    std::string spk2utt_rspecifier;
    std::string wav_rspecifier;
    std::string rec_wspecifier;
    std::string silence_phones;

    char res_str[1000];
    int32 app_chunksize;
    bool err_flag;
    char fst_addr_str[30];
    char word_syms_addr_str[30];
    char feature_info_addr_str[30];
    char am_nnet_addr_str[30];
};


#endif //KALDI_FAI_DECODER_WRAPPER_H
