//
// Created by uk on 2022/11/10.
//
#include "fai-decoder-wrapper.h"


FaiDecoderWrapper::~FaiDecoderWrapper() {
    delete od; // faidecoder 객체를 new로 생성했으므로 삭제한다.
    od = NULL;
}


void FaiDecoderWrapper::decode_finalize(char *transactionid) {
    try {
        string utt = "stream";
        int64 num_frames = 0;
        double tot_like = 0.0;
        CompactLattice clat;
        bool end_of_utterance = true;

        // string log_file_name(transactionid);
        // log_file_name = "./logs/kaldi/" + log_file_name + ".log";
        // reopen(log_file_name.c_str(), "a", stderr);

        od->FinalizeDecoding();

        if (align_wxfilename == "") {
            od->GetRecognitionResult(end_of_utterance, lm_scale, wip, utt, clat,
                                     &num_frames, &tot_like, deb_lv, nbest, pair_out, detokenized_flag, confBR_flag);
        } else {
            od->GetRecognitionResult(end_of_utterance, lm_scale, wip, utt, clat,
                                     &num_frames, &tot_like, deb_lv, nbest, pair_out, pair_word_align_out,
                                     pair_phone_align_out, detokenized_flag, confBR_flag);
        }
        od->GetStateInfo();

        strcpy(res_str, (pair_out[0].second).c_str());

        // if (!err_flag)
        //{
        //   remove(log_file_name.c_str());
        // }
    }
    catch (exception e) {
        std::cerr << "[" << transactionid << "] Decoding finalize Error" << endl;
        strcpy(res_str, "Decoding finalize Error.(Kaldi library)");
    }
}

void FaiDecoderWrapper::decode_finalize_manual(char *transactionid) {
    try {
        string utt = "stream";
        int64 num_frames = 0;
        double tot_like = 0.0;
        CompactLattice clat;
        bool end_of_utterance = true;
        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        // string log_file_name(transactionid);
        // log_file_name = "./logs/kaldi/" + log_file_name + ".log";
        // reopen(log_file_name.c_str(), "a", stderr);

        od->InputFinished();
        od->SilenceWeightUpdate(delta_weights);
        od->AdvanceDecoding();
        od->FinalizeDecoding();

        if (align_wxfilename == "") {
            od->GetRecognitionResult(end_of_utterance, lm_scale, wip, utt, clat,
                                     &num_frames, &tot_like, deb_lv, nbest, pair_out, detokenized_flag, confBR_flag);
        } else {
            od->GetRecognitionResult(end_of_utterance, lm_scale, wip, utt, clat,
                                     &num_frames, &tot_like, deb_lv, nbest, pair_out, pair_word_align_out,
                                     pair_phone_align_out, detokenized_flag, confBR_flag);
        }
        od->GetStateInfo();

        strcpy(res_str, (pair_out[0].second).c_str());

        // if (!err_flag)
        //{
        //   remove(log_file_name.c_str());
        // }
    }
    catch (exception e) {
        std::cerr << "[" << transactionid << "] Decoding finalize Error" << endl;
        strcpy(res_str, "Decoding finalize Error.(Kaldi library)");
    }
}

void FaiDecoderWrapper::run_decode_stream(char *buffer, int buffer_size, char *transactionid) {
    try {
        double tot_like = 0.0;
        int64 num_frames = 0;
        string utt = "stream";

        // string log_file_name(transactionid);
        // log_file_name = "./logs/kaldi/" + log_file_name + ".log";
        // freopen(log_file_name.c_str(), "w", stderr);

        membuf sbuf(buffer, buffer + buffer_size);
        std::istream is(&sbuf);
        WaveData wave_data;
        wave_data.Read(is);

        SubVector<BaseFloat> data(wave_data.Data(), 0);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0)
                chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                           : samp_remaining;

            SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);

            od->AcceptWaveform(samp_freq, wave_part);

            samp_offset += num_samp;

            if (buffer_size < app_chunksize + 44) { // Add Header size, Test 용 if 절, 파일을 사용하기 때문.
                // no more input. flush out last frames
                od->InputFinished();

                od->SilenceWeightUpdate(delta_weights);
                od->AdvanceDecoding();
                wave_part.~SubVector();

                data.~SubVector();
                wave_data.Clear();
                is.clear();
                strcpy(res_str, "streamEnd");
                return;
            }

            od->SilenceWeightUpdate(delta_weights);
            od->AdvanceDecoding();
            wave_part.~SubVector();

            if (do_endpointing && od->EndpointDetected()) {
                od->InputFinished();

                od->SilenceWeightUpdate(delta_weights);
                od->AdvanceDecoding();
                wave_part.~SubVector();

                data.~SubVector();
                wave_data.Clear();
                is.clear();
                strcpy(res_str, "endDeteced");
                return;
            }
        }
        data.~SubVector();
        wave_data.Clear();
        is.clear();
        strcpy(res_str, "Decoding");
    }
    catch (exception e) {
        std::cerr << "[" << transactionid << "] Decoding Error" << endl;
        strcpy(res_str, "Decoding Error.(Kaldi library)");
        err_flag = true;
    }
}

void FaiDecoderWrapper::run_decode_file(char *buffer, int buffer_size, char *transactionid) {
    try {
        double tot_like = 0.0;
        int64 num_frames = 0;
        string utt = "file";

        // string log_file_name(transactionid);
        // log_file_name = "./logs/kaldi/" + log_file_name + ".log";
        // freopen(log_file_name.c_str(), "w", stderr);

        membuf sbuf(buffer, buffer + buffer_size);
        std::istream is(&sbuf);
        WaveData wave_data;
        wave_data.Read(is);

        SubVector<BaseFloat> data(wave_data.Data(), 0);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0)
                chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                           : samp_remaining;

            SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);

            od->AcceptWaveform(samp_freq, wave_part);

            samp_offset += num_samp;

            if (samp_remaining < app_chunksize) { // Add Header size, Test 용 if 절, 파일을 사용하기 때문.
                // no more input. flush out last frames
                od->InputFinished();

                od->SilenceWeightUpdate(delta_weights);
                od->AdvanceDecoding();
                wave_part.~SubVector();

                data.~SubVector();
                wave_data.Clear();
                is.clear();
                strcpy(res_str, "fileEnd");
                return;
            }

            od->SilenceWeightUpdate(delta_weights);
            od->AdvanceDecoding();
            wave_part.~SubVector();

            if (do_endpointing && od->EndpointDetected()) {
                od->InputFinished();

                od->SilenceWeightUpdate(delta_weights);
                od->AdvanceDecoding();
                wave_part.~SubVector();

                data.~SubVector();
                wave_data.Clear();
                is.clear();
                strcpy(res_str, "endDeteced");
                return;
            }
        }
        data.~SubVector();
        wave_data.Clear();
        is.clear();
        strcpy(res_str, "Decoding");
    }
    catch (exception e) {
        std::cerr << "[" << transactionid << "] Decoding Error" << endl;
        strcpy(res_str, "Decoding Error.(Kaldi library)");
        err_flag = true;
    }
}

void FaiDecoderWrapper::run_decode_websocket(char *buffer, int buffer_size, char *transactionid) {
    try {
        double tot_like = 0.0;
        int64 num_frames = 0;
        string utt = "websocket";

        // string log_file_name(transactionid);
        // log_file_name = "./logs/kaldi/" + log_file_name + ".log";
        // freopen(log_file_name.c_str(), "a", stderr);

        membuf sbuf(buffer, buffer + buffer_size);
        std::istream is(&sbuf);
        WaveData wave_data;
        wave_data.Read(is);

        SubVector<BaseFloat> data(wave_data.Data(), 0);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0)
                chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                           : samp_remaining;

            SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);

            od->AcceptWaveform(samp_freq, wave_part);

            samp_offset += num_samp;

            od->SilenceWeightUpdate(delta_weights);
            od->AdvanceDecoding();
            wave_part.~SubVector();
        }

        CompactLattice clat;

        strcpy(res_str, (od->GetRecognitionPartialResult(lm_scale, wip, utt, clat,
                                                         &num_frames, &tot_like, deb_lv, pair_out, detokenized_flag))
                .c_str());
        data.~SubVector();
        wave_data.Clear();
        is.clear();
    }
    catch (exception e) {
        std::cerr << "[" << transactionid << "] Decoding Error" << endl;
        strcpy(res_str, "Decoding Error.(Kaldi library)");
        err_flag = true;
    }
}

void FaiDecoderWrapper::run_decode_websocket_end(char *buffer, int buffer_size, char *transactionid) {
    try {
        double tot_like = 0.0;
        int64 num_frames = 0;
        string utt = "websocket_end";

        // string log_file_name(transactionid);
        // log_file_name = "./logs/kaldi/" + log_file_name + ".log";
        // freopen(log_file_name.c_str(), "a", stderr);

        membuf sbuf(buffer, buffer + buffer_size);
        std::istream is(&sbuf);
        WaveData wave_data;
        wave_data.Read(is);

        SubVector<BaseFloat> data(wave_data.Data(), 0);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0)
                chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat>> delta_weights;

        while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                           : samp_remaining;

            SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);

            od->AcceptWaveform(samp_freq, wave_part);

            samp_offset += num_samp;

            if (buffer_size < app_chunksize) {
                od->InputFinished();

                od->SilenceWeightUpdate(delta_weights);
                od->AdvanceDecoding();
                wave_part.~SubVector();

                data.~SubVector();
                wave_data.Clear();
                is.clear();

                return;
            }

            od->SilenceWeightUpdate(delta_weights);
            od->AdvanceDecoding();
            wave_part.~SubVector();
        }

        data.~SubVector();
        wave_data.Clear();
        is.clear();
    }
    catch (exception e) {
        std::cerr << "[" << transactionid << "] Decoding Error" << endl;
        strcpy(res_str, "Decoding Error.(Kaldi library)");
        err_flag = true;
    }
}

char *FaiDecoderWrapper::get_result() {
    return res_str;
}

// faidecoder 객체를 inference 수행하기 전 상태로 reset 한다.
void FaiDecoderWrapper::reset_onlinedecoder() {
    od->Reset();
    pair_out.clear();
    pair_word_align_out.clear();
    pair_phone_align_out.clear();
    err_flag = false;
}

char *FaiDecoderWrapper::get_fst_addr_string() {
    return fst_addr_str;
}

char *FaiDecoderWrapper::get_word_syms_addr_string() {
    return word_syms_addr_str;
}

char *FaiDecoderWrapper::get_feature_info_addr_string() {
    return feature_info_addr_str;
}

char *FaiDecoderWrapper::get_am_nnet_addr_string() {
    return am_nnet_addr_str;
}

fst::Fst<fst::StdArc> *FaiDecoderWrapper::get_fst_addr() {
    return od->get_fst_addr();
}

fst::SymbolTable *FaiDecoderWrapper::get_word_syms_addr() {
    return od->get_word_syms_addr();
}

OnlineNnet2FeaturePipelineInfo *FaiDecoderWrapper::get_feature_info_addr() {
    return od->get_fearute_info_addr();
}

nnet3::AmNnetSimple *FaiDecoderWrapper::get_am_nnet_addr() {
    return od->get_am_nnet_addr();
}

float FaiDecoderWrapper::get_am_score() {
    return od->getAmScore();
}

double FaiDecoderWrapper::get_lm_score() {
    return od->getLmScore();
}

void FaiDecoderWrapper::get_fst_addr_to_string() {
    std::stringstream ss;
    ss << get_fst_addr();
    const char *c_addr = ss.str().c_str();
    strcpy(fst_addr_str, c_addr);
}

void FaiDecoderWrapper::get_word_syms_addr_to_string() {
    std::stringstream ss;
    ss << get_word_syms_addr();
    const char *c_addr = ss.str().c_str();
    strcpy(word_syms_addr_str, c_addr);
}

void FaiDecoderWrapper::get_feature_info_addr_to_string() {
    std::stringstream ss;
    ss << get_feature_info_addr();
    const char *c_addr = ss.str().c_str();
    strcpy(feature_info_addr_str, c_addr);
}

void FaiDecoderWrapper::get_am_nnet_addr_to_string() {
    std::stringstream ss;
    ss << get_am_nnet_addr();
    const char *c_addr = ss.str().c_str();
    strcpy(am_nnet_addr_str, c_addr);
}

fst::Fst<fst::StdArc> *FaiDecoderWrapper::get_fst_addr_to_pointer(char *addr_str) {
    std::stringstream ss(addr_str);
    void *p = nullptr;
    ss >> p;
    ss.clear();
    return (fst::Fst<fst::StdArc> *) p;
}

fst::SymbolTable *FaiDecoderWrapper::get_word_syms_addr_to_pointer(char *addr_str) {
    std::stringstream ss(addr_str);
    void *p = nullptr;
    ss >> p;
    ss.clear();
    return (fst::SymbolTable *) p;
}

OnlineNnet2FeaturePipelineInfo *FaiDecoderWrapper::get_feature_info_addr_to_pointer(char *addr_str) {
    std::stringstream ss(addr_str);
    void *p = nullptr;
    ss >> p;
    ss.clear();
    return (OnlineNnet2FeaturePipelineInfo *) p;
}

nnet3::AmNnetSimple *FaiDecoderWrapper::get_am_nnet_addr_to_pointer(char *addr_str) {
    std::stringstream ss(addr_str);
    void *p = nullptr;
    ss >> p;
    ss.clear();
    return (nnet3::AmNnetSimple *) p;
}

// faidecoder 객체를 생성 한다.
void FaiDecoderWrapper::load_onlinedecoder() {
    od = new FaiDecoder(feature_opts, decodable_opts, decoder_opts, wb_info_opts, endpoint_opts, online,
                        nnet3_rxfilename,
                        fst_rxfilename, word_syms_rxfilename, word_boundary_rxfilename, phone_syms_rxfilename,
                        align_wxfilename, chunk_length_secs);
}

void FaiDecoderWrapper::load_onlinedecoder_addr(char *addr_fst, char *addr_word_syms, char *addr_feature_info,
                                                char *addr_am_nnet) {
    fst::Fst<fst::StdArc> *fst_addr = get_fst_addr_to_pointer(addr_fst);
    fst::SymbolTable *word_syms_addr = get_word_syms_addr_to_pointer(addr_word_syms);
    OnlineNnet2FeaturePipelineInfo *feature_info_addr = get_feature_info_addr_to_pointer(addr_feature_info);
    nnet3::AmNnetSimple *am_nnet_addr = get_am_nnet_addr_to_pointer(addr_am_nnet);
    od = new FaiDecoder(feature_opts, decodable_opts, decoder_opts, wb_info_opts, endpoint_opts, online,
                        nnet3_rxfilename,
                        fst_rxfilename, word_syms_rxfilename, word_boundary_rxfilename, phone_syms_rxfilename,
                        align_wxfilename, chunk_length_secs, fst_addr, word_syms_addr, feature_info_addr,
                        am_nnet_addr);
}

//동작에 필요한 config 값들을 초기화 한다.
//해당 값들을 confiure.ini등의 파일로 설정 할 수 있도록 해야한다.
void FaiDecoderWrapper::init_configure(int _app_chunksize,
                                       float _chunk_length_secs, bool _do_endpointing, bool _detokenized_flag,
                                       bool _confBR_flag,
                                       int _nbest, float _lm_scale, float _wip, char *_silence_phones) {
    app_chunksize = _app_chunksize;
    word_syms_rxfilename = "";
    align_wxfilename = "";
    word_boundary_rxfilename = "";
    phone_syms_rxfilename = "";
    chunk_length_secs = _chunk_length_secs;
    do_endpointing = _do_endpointing;
    online = true;
    detokenized_flag = _detokenized_flag;
    confBR_flag = _confBR_flag;
    deb_lv = 0;
    nbest = _nbest;
    lm_scale = _lm_scale;
    wip = _wip;
    silence_phones = _silence_phones;

    const char *usage =
            "Usage: online-decoder [options] <nnet3-in> <fst-in> "
            "<wav-rspecifier> <rec-wspecifier>\n";

    ParseOptions po(usage);
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("align-wxfilename", &align_wxfilename,
                "align output filename");
    po.Register("phone-symbol-table", &phone_syms_rxfilename,
                "phone symbol table filename");
    po.Register("word-boundary-rxfilename", &word_boundary_rxfilename,
                "word boundary input filename");
    po.Register("lm-scale", &lm_scale,
                "language model scale factor");
    po.Register("word-ins-penalty", &wip,
                "word insert penalty factor");
    po.Register("deb-lv", &deb_lv,
                "debugging level");
    po.Register("detokenized-flag", &detokenized_flag,
                "If true, calculate and print out confidence score and bayesian risk");
    po.Register("confBR-flag", &confBR_flag,
                "If true, MBR Decoding and print out confidence score");
    po.Register("nbest", &nbest,
                "nbest size");
    po.Register("endpoint.do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);
    wb_info_opts.Register(&po);
    endpoint_opts.silence_phones = silence_phones;

    //아래는 shell script로 실행시 argument로 입력 받던 부분으로 수정이 필요하다.
    //현재는 동작확인을 위해 지정된 값을 할당
    char *tmp[9] = {
            "fai-decoder",
            "--verbose=1",
            "--config=/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/conf/decode.conf",
            "--mfcc-config=/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/conf/mfcc_hires.conf",
            "--ivector-extraction-config=/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/conf/ivector_extractor.conf",
            "--word-symbol-table=/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/models/graph/words.txt",
            "--phone-symbol-table=/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/models/graph/phones.txt",
            "--word-boundary-rxfilename=/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/models/graph/phones/word_boundary.int",
            "--align-wxfilename=align-wxfilename"};
    po.Read(9, tmp);

    // if (po.NumArgs() != 2) {
    //   po.PrintUsage();
    //   return;
    // }

    nnet3_rxfilename = "/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/models/final.mdl";
    fst_rxfilename = "/Users/uk/CLionProjects/fai-kaldi/src/faidecoderbin/rsc/models/graph/HCLG.fst";

    err_flag = false;
}

// jna는 c++을 지원하지 않기 때문에 extern "C"를 통해 C처럼 사용 할 수 있게 한다.
extern "C" {
FaiDecoderWrapper *create_odw() {
    return new FaiDecoderWrapper();
}

void init_configure(FaiDecoderWrapper *f, int _app_chunksize,
                    float _chunk_length_secs, bool _do_endpointing, bool _detokenized_flag, bool _confBR_flag,
                    int _nbest, float _lm_scale, float _wip, char *_silence_phones) {
    f->init_configure(_app_chunksize,
                      _chunk_length_secs, _do_endpointing, _detokenized_flag, _confBR_flag,
                      _nbest, _lm_scale, _wip, _silence_phones);
}

void load_onlinedecoder(FaiDecoderWrapper *f) {
    f->load_onlinedecoder();
}

void load_onlinedecoder_addr(FaiDecoderWrapper *f, char *addr_fst, char *addr_word_syms, char *addr_feature_info,
                             char *addr_am_nnet) {
    f->load_onlinedecoder_addr(addr_fst, addr_word_syms, addr_feature_info, addr_am_nnet);
}

char *get_fst_addr(FaiDecoderWrapper *f) {
    f->get_fst_addr_to_string();
    return f->get_fst_addr_string();
}

char *get_word_syms_addr(FaiDecoderWrapper *f) {
    f->get_word_syms_addr_to_string();
    return f->get_word_syms_addr_string();
}

char *get_feature_info_addr(FaiDecoderWrapper *f) {
    f->get_feature_info_addr_to_string();
    return f->get_feature_info_addr_string();
}

char *get_am_nnet_addr(FaiDecoderWrapper *f) {
    f->get_am_nnet_addr_to_string();
    return f->get_am_nnet_addr_string();
}

void reset_onlinedecoder(FaiDecoderWrapper *f) {
    f->reset_onlinedecoder();
}

float get_am_score(FaiDecoderWrapper *f) {
    return f->get_am_score();
}

float get_lm_score(FaiDecoderWrapper *f) {
    return (float) f->get_lm_score();
}

char *run_decode_stream(FaiDecoderWrapper *f, char *buffer, int buffer_size, char *transactionid) {
    f->run_decode_stream(buffer, buffer_size, transactionid);
    return f->get_result();
}

char *run_decode_file(FaiDecoderWrapper *f, char *buffer, int buffer_size, char *transactionid) {
    f->run_decode_file(buffer, buffer_size, transactionid);
    return f->get_result();
}

char *run_decode_websocket(FaiDecoderWrapper *f, char *buffer, int buffer_size, char *transactionid) {
    f->run_decode_websocket(buffer, buffer_size, transactionid);
    return f->get_result();
}

void run_decode_websocket_end(FaiDecoderWrapper *f, char *buffer, int buffer_size, char *transactionid) {
    f->run_decode_websocket_end(buffer, buffer_size, transactionid);
}

char *run_finalize(FaiDecoderWrapper *f, char *transactionid) {
    f->decode_finalize(transactionid);
    return f->get_result();
}

char *run_finalize_manual(FaiDecoderWrapper *f, char *transactionid) {
    f->decode_finalize_manual(transactionid);
    return f->get_result();
}

}
