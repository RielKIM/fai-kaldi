#include "fai-decoder.h"

namespace kaldi
{
    FaiDecoder::FaiDecoder(const OnlineNnet2FeaturePipelineConfig &feature_opts,
                           const nnet3::NnetSimpleLoopedComputationOptions &decodable_opts,
                           const LatticeFasterDecoderConfig &decoder_opts, const WordBoundaryInfoNewOpts wordboundary_info_opts,
                           const OnlineEndpointConfig &endpoint_opts,
                           const bool online, const std::string nnet3_rxfilename, const std::string fst_rxfilename,
                           const std::string word_syms_rxfilename, const std::string word_boundary_rxfilename,
                           const std::string phone_syms_rxfilename,
                           const std::string ctm_wxfilename,
                           const BaseFloat chunk_length_secs) : feature_opts_(feature_opts), decoder_opts_(decoder_opts), wordboundary_info_opts_(wordboundary_info_opts),
                                                                      decodable_opts_(decodable_opts), endpoint_opts_(endpoint_opts), online_(online), nnet3_rxfilename_(nnet3_rxfilename),
                                                                      fst_rxfilename_(fst_rxfilename), word_syms_rxfilename_(word_syms_rxfilename), phone_syms_rxfilename_(phone_syms_rxfilename),
                                                                      word_boundary_rxfilename_(word_boundary_rxfilename),
                                                                      ctm_wxfilename_(ctm_wxfilename), chunk_length_secs_(chunk_length_secs)
    {
        Init();
    }

    FaiDecoder::FaiDecoder(const OnlineNnet2FeaturePipelineConfig &feature_opts,
                           const nnet3::NnetSimpleLoopedComputationOptions &decodable_opts,
                           const LatticeFasterDecoderConfig &decoder_opts, const WordBoundaryInfoNewOpts wordboundary_info_opts,
                           const OnlineEndpointConfig &endpoint_opts,
                           const bool online, const std::string nnet3_rxfilename, const std::string fst_rxfilename,
                           const std::string word_syms_rxfilename, const std::string word_boundary_rxfilename,
                           const std::string phone_syms_rxfilename,
                           const std::string ctm_wxfilename, const BaseFloat chunk_length_secs,
                           fst::Fst<fst::StdArc> *fst_addr,
                           fst::SymbolTable *word_syms_addr, OnlineNnet2FeaturePipelineInfo* feature_info_addr,
                           nnet3::AmNnetSimple *am_nnet_addr) : feature_opts_(feature_opts), decoder_opts_(decoder_opts), wordboundary_info_opts_(wordboundary_info_opts),
                                                                     decodable_opts_(decodable_opts), endpoint_opts_(endpoint_opts), online_(online), nnet3_rxfilename_(nnet3_rxfilename),
                                                                     fst_rxfilename_(fst_rxfilename), word_syms_rxfilename_(word_syms_rxfilename), phone_syms_rxfilename_(phone_syms_rxfilename),
                                                                     word_boundary_rxfilename_(word_boundary_rxfilename),
                                                                     ctm_wxfilename_(ctm_wxfilename), chunk_length_secs_(chunk_length_secs)
    {
        Init(fst_addr, word_syms_addr, feature_info_addr, am_nnet_addr);
    }

    void FaiDecoder::LoadModel(fst::Fst<fst::StdArc> *fst_addr, fst::SymbolTable *word_syms_addr, nnet3::AmNnetSimple *am_nnet_addr)
    {
        bool binary;

        if (feature_opts_.global_cmvn_stats_rxfilename != "")
        {
            ReadKaldiObject(feature_opts_.global_cmvn_stats_rxfilename,
                            &global_cmvn_stats_);
        }

        Input ki(nnet3_rxfilename_, &binary);
        trans_model_.Read(ki.Stream(), binary);
        am_nnet_ = am_nnet_addr;

        // SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
        // SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
        // nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet_.GetNnet()));
        SetBatchnormTestMode(true, &(am_nnet_->GetNnet()));
        SetDropoutTestMode(true, &(am_nnet_->GetNnet()));
        nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet_->GetNnet()));


        if (word_boundary_rxfilename_ != "")
        {
            wordboudary_info_ = new WordBoundaryInfo(wordboundary_info_opts_, word_boundary_rxfilename_);
        }

        decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts_,
                                                                   am_nnet_);

        decode_fst_ = fst_addr;
        word_syms_ = word_syms_addr;
        phone_syms_ = NULL;

        // if (word_syms_rxfilename_ != "")
        //         if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename_)))
        //                 KALDI_ERR << "Could not read symbol table from file "
        //                 << word_syms_rxfilename_;

        if (phone_syms_rxfilename_ != "")
            if (!(phone_syms_ = fst::SymbolTable::ReadText(phone_syms_rxfilename_)))
                KALDI_ERR << "Could not read symbol table from file "
                          << phone_syms_rxfilename_;
    }

    void FaiDecoder::LoadModel()
    {
        bool binary;

        if (feature_opts_.global_cmvn_stats_rxfilename != "")
        {
            ReadKaldiObject(feature_opts_.global_cmvn_stats_rxfilename,
                            &global_cmvn_stats_);
        }

        Input ki(nnet3_rxfilename_, &binary);
        trans_model_.Read(ki.Stream(), binary);
        am_nnet_ = new nnet3::AmNnetSimple();
        am_nnet_->Read(ki.Stream(), binary);
        SetBatchnormTestMode(true, &(am_nnet_->GetNnet()));
        SetDropoutTestMode(true, &(am_nnet_->GetNnet()));
        nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet_->GetNnet()));

        if (word_boundary_rxfilename_ != "")
        {
            wordboudary_info_ = new WordBoundaryInfo(wordboundary_info_opts_, word_boundary_rxfilename_);
        }

        decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts_,
                                                                   am_nnet_);

        decode_fst_ = fst::ReadFstKaldiGeneric(fst_rxfilename_);

        word_syms_ = NULL;
        phone_syms_ = NULL;

        if (word_syms_rxfilename_ != "")
            if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename_)))
                KALDI_ERR << "Could not read symbol table from file "
                          << word_syms_rxfilename_;

        if (phone_syms_rxfilename_ != "")
            if (!(phone_syms_ = fst::SymbolTable::ReadText(phone_syms_rxfilename_)))
                KALDI_ERR << "Could not read symbol table from file "
                          << phone_syms_rxfilename_;
    }

    void FaiDecoder::Init(fst::Fst<fst::StdArc> *fst_addr, fst::SymbolTable *word_syms_addr, OnlineNnet2FeaturePipelineInfo* feature_info_addr, nnet3::AmNnetSimple *am_nnet_addr)
    {
        frame_shift_ = 0.03;
        feature_info_ = feature_info_addr;

        if (!online_)
        {
            // feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
            // feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;
            chunk_length_secs_ = -1.0;
        }

        LoadModel(fst_addr, word_syms_addr, am_nnet_addr);

        adaptation_state_ = new OnlineIvectorExtractorAdaptationState(feature_info_->ivector_extractor_info);
        cmvn_state_ = new OnlineCmvnState(global_cmvn_stats_);

        feature_pipeline_ = new OnlineNnet2FeaturePipeline(*feature_info_);

        feature_pipeline_->SetAdaptationState(*adaptation_state_);
        feature_pipeline_->SetCmvnState(*cmvn_state_);

        silence_weighting_ = new OnlineSilenceWeighting(trans_model_,
                                                        feature_info_->silence_weighting_config,
                                                        decodable_opts_.frame_subsampling_factor);

        KALDI_LOG << "decoder_opts:" << decoder_opts_.lattice_beam;
        decoder_ = new SingleUtteranceNnet3Decoder(decoder_opts_, trans_model_,
                                                   *decodable_info_,
                                                   *decode_fst_, feature_pipeline_);
    }

    void FaiDecoder::Init()
    {
        frame_shift_ = 0.03;
        feature_info_ = new OnlineNnet2FeaturePipelineInfo(feature_opts_);

        if (!online_)
        {
            feature_info_->ivector_extractor_info.use_most_recent_ivector = true;
            feature_info_->ivector_extractor_info.greedy_ivector_extractor = true;
            chunk_length_secs_ = -1.0;
        }

        LoadModel();

        adaptation_state_ = new OnlineIvectorExtractorAdaptationState(feature_info_->ivector_extractor_info);
        cmvn_state_ = new OnlineCmvnState(global_cmvn_stats_);

        feature_pipeline_ = new OnlineNnet2FeaturePipeline(*feature_info_);

        feature_pipeline_->SetAdaptationState(*adaptation_state_);
        feature_pipeline_->SetCmvnState(*cmvn_state_);

        silence_weighting_ = new OnlineSilenceWeighting(trans_model_,
                                                        feature_info_->silence_weighting_config,
                                                        decodable_opts_.frame_subsampling_factor);

        KALDI_LOG << "decoder_opts:" << decoder_opts_.lattice_beam;
        decoder_ = new SingleUtteranceNnet3Decoder(decoder_opts_, trans_model_,
                                                   *decodable_info_,
                                                   *decode_fst_, feature_pipeline_);
    }

    void FaiDecoder::Reset()
    {
        if (decoder_)
            delete decoder_;
        if (silence_weighting_)
            delete silence_weighting_;
        if (feature_pipeline_)
            delete feature_pipeline_;
        if (cmvn_state_)
            delete cmvn_state_;
        if (adaptation_state_)
            delete adaptation_state_;

        adaptation_state_ = new OnlineIvectorExtractorAdaptationState(feature_info_->ivector_extractor_info);
        cmvn_state_ = new OnlineCmvnState(global_cmvn_stats_);

        feature_pipeline_ = new OnlineNnet2FeaturePipeline(*feature_info_);

        feature_pipeline_->SetAdaptationState(*adaptation_state_);
        feature_pipeline_->SetCmvnState(*cmvn_state_);

        silence_weighting_ = new OnlineSilenceWeighting(trans_model_,
                                                        feature_info_->silence_weighting_config,
                                                        decodable_opts_.frame_subsampling_factor);

        decoder_ = new SingleUtteranceNnet3Decoder(decoder_opts_, trans_model_,
                                                   *decodable_info_,
                                                   *decode_fst_, feature_pipeline_);
    }

    void FaiDecoder::SilenceWeightUpdate(std::vector<std::pair<int32, BaseFloat>> delta_weights)
    {
        if (silence_weighting_->Active() &&
            feature_pipeline_->IvectorFeature() != NULL)
        {
            silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder());
            silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(),
                                                &delta_weights);
            feature_pipeline_->IvectorFeature()->UpdateFrameWeights(delta_weights);
        }
    }

    void FaiDecoder::AdvanceDecoding()
    {
        decoder_->AdvanceDecoding();
    }

    bool FaiDecoder::EndpointDetected()
    {
        return decoder_->EndpointDetected(endpoint_opts_);
    }

    void FaiDecoder::FinalizeDecoding()
    {
        decoder_->FinalizeDecoding();
    }

    void FaiDecoder::InputFinished()
    {
        feature_pipeline_->InputFinished();
    }

    void FaiDecoder::AcceptWaveform(BaseFloat samp_freq, SubVector<BaseFloat> wave_part)
    {
        feature_pipeline_->AcceptWaveform(samp_freq, wave_part);
    }

    void FaiDecoder::GetStateInfo()
    {
        feature_pipeline_->GetAdaptationState(adaptation_state_);
        feature_pipeline_->GetCmvnState(cmvn_state_);
    }

    void FaiDecoder::GetRecognitionPartialResult(OnlineTimer &decoding_timer, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                                                 CompactLattice &clat,
                                                 int64 *tot_num_frames,
                                                 double *tot_like, int32 deb_lv,
                                                 std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag)
    {
        if (decoder_->NumFramesDecoded() == 0)
            return;

        KALDI_VLOG(4) << "internal timer_1:" << decoding_timer.Elapsed();
        decoder_->GetLattice(false, &clat);
        KALDI_VLOG(4) << "internal timer_2:" << decoding_timer.Elapsed();

        KALDI_VLOG(4) << "internal timer_3:" << decoding_timer.Elapsed();
        ApplyLatticeScale(lm_scale, 1.0, clat);
        KALDI_VLOG(4) << "internal timer_4:" << decoding_timer.Elapsed();
        AddWordInsPenToCompactLattice(wip, &clat);
        KALDI_VLOG(4) << "internal timer_5:" << decoding_timer.Elapsed();

        PartialGetPrintInfo(decoding_timer, utt, clat, tot_num_frames, tot_like, deb_lv, pair_out, detokenized_flag);
    }

    string FaiDecoder::GetRecognitionPartialResult(const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                                                   CompactLattice &clat,
                                                   int64 *tot_num_frames,
                                                   double *tot_like, int32 deb_lv,
                                                   std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag)
    {
        if (decoder_->NumFramesDecoded() == 0)
            return "";

        decoder_->GetLattice(false, &clat);
        ApplyLatticeScale(lm_scale, 1.0, clat);
        AddWordInsPenToCompactLattice(wip, &clat);

        return PartialGetPrintInfo(utt, clat, tot_num_frames, tot_like, deb_lv, pair_out, detokenized_flag);
    }

    void FaiDecoder::GetRecognitionResult(OnlineTimer &decoding_timer, const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                                          CompactLattice &clat,
                                          int64 *tot_num_frames,
                                          double *tot_like, int32 deb_lv, int32 nbest_size,
                                          std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag, bool confBR_flag)
    {
        KALDI_VLOG(4) << "internal timer_1:" << decoding_timer.Elapsed();
        decoder_->GetLattice(end_of_utterance, &clat);
        KALDI_VLOG(4) << "internal timer_2:" << decoding_timer.Elapsed();

        KALDI_VLOG(4) << "internal timer_3:" << decoding_timer.Elapsed();
        ApplyLatticeScale(lm_scale, 1.0, clat);
        KALDI_VLOG(4) << "internal timer_4:" << decoding_timer.Elapsed();
        AddWordInsPenToCompactLattice(wip, &clat);
        KALDI_VLOG(4) << "internal timer_5:" << decoding_timer.Elapsed();

        GetPrintInfo(decoding_timer, utt, clat, tot_num_frames, tot_like, deb_lv, nbest_size, pair_out, detokenized_flag);

        if (confBR_flag)
        {
            std::pair<BaseFloat, double> pair_temp;
            CompactLattice prune_clat(clat);
            PruneLatticeW(decoder_opts_.lattice_beam, prune_clat);
            GetConfBR(prune_clat, pair_temp, utt);
        }
    }

    void FaiDecoder::GetRecognitionResult(const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                                          CompactLattice &clat,
                                          int64 *tot_num_frames,
                                          double *tot_like, int32 deb_lv, int32 nbest_size,
                                          std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag, bool confBR_flag)
    {
        decoder_->GetLattice(end_of_utterance, &clat);
        ApplyLatticeScale(lm_scale, 1.0, clat);
        AddWordInsPenToCompactLattice(wip, &clat);

        GetPrintInfo(utt, clat, tot_num_frames, tot_like, deb_lv, nbest_size, pair_out, detokenized_flag);

        if (confBR_flag)
        {
            std::pair<BaseFloat, double> pair_temp;
            CompactLattice prune_clat(clat);
            PruneLatticeW(decoder_opts_.lattice_beam, prune_clat);
            GetConfBR(prune_clat, pair_temp, utt);
        }
    }

    void FaiDecoder::GetRecognitionResult(OnlineTimer &decoding_timer, const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                                          CompactLattice &clat,
                                          int64 *tot_num_frames,
                                          double *tot_like, int32 deb_lv, int32 nbest_size,
                                          std::vector<std::pair<std::string, std::string>> &pair_out,
                                          std::vector<std::pair<std::string, std::string>> &pair_word_align_out,
                                          std::vector<std::pair<std::string, std::string>> &pair_phone_align_out,
                                          bool detokenized_flag, bool confBR_flag)
    {
        KALDI_VLOG(4) << "internal timer_1:" << decoding_timer.Elapsed();
        decoder_->GetLattice(end_of_utterance, &clat);
        KALDI_VLOG(4) << "internal timer_2:" << decoding_timer.Elapsed();

        KALDI_VLOG(4) << "internal timer_3:" << decoding_timer.Elapsed();
        ApplyLatticeScale(lm_scale, 1.0, clat);
        KALDI_VLOG(4) << "internal timer_4:" << decoding_timer.Elapsed();
        AddWordInsPenToCompactLattice(wip, &clat);
        KALDI_VLOG(4) << "internal timer_5:" << decoding_timer.Elapsed();

        GetPrintInfo(decoding_timer, utt, clat, tot_num_frames, tot_like, deb_lv, pair_out, pair_word_align_out, pair_phone_align_out, detokenized_flag);

        if (confBR_flag)
        {
            std::pair<BaseFloat, double> pair_temp;
            CompactLattice prune_clat(clat);
            PruneLatticeW(decoder_opts_.lattice_beam, prune_clat);
            GetConfBR(prune_clat, pair_temp, utt);
        }
    }

    void FaiDecoder::GetRecognitionResult(const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                                          CompactLattice &clat,
                                          int64 *tot_num_frames,
                                          double *tot_like, int32 deb_lv, int32 nbest_size,
                                          std::vector<std::pair<std::string, std::string>> &pair_out,
                                          std::vector<std::pair<std::string, std::string>> &pair_word_align_out,
                                          std::vector<std::pair<std::string, std::string>> &pair_phone_align_out,
                                          bool detokenized_flag, bool confBR_flag)
    {
        decoder_->GetLattice(end_of_utterance, &clat);
        ApplyLatticeScale(lm_scale, 1.0, clat);
        AddWordInsPenToCompactLattice(wip, &clat);

        GetPrintInfo(utt, clat, tot_num_frames, tot_like, deb_lv, pair_out, pair_word_align_out, pair_phone_align_out, detokenized_flag);

        if (confBR_flag)
        {
            std::pair<BaseFloat, double> pair_temp;
            CompactLattice prune_clat(clat);
            PruneLatticeW(decoder_opts_.lattice_beam, prune_clat);
            GetConfBR(prune_clat, pair_temp, utt);
        }
    }

    void FaiDecoder::PruneLatticeW(BaseFloat beam, CompactLattice &prune_clat)
    {
        if (!PruneLattice(beam, &prune_clat))
        {
            KALDI_WARN << "Error pruning lattice";
        }
    }

    void FaiDecoder::GetConfBR(const CompactLattice &clat, std::pair<BaseFloat, double> &pair_temp, const std::string utt)
    {
        MinimumBayesRisk mbr(clat);
        std::string one_best_string;

        pair_temp.first = GetAvgConf(mbr.GetOneBestConfidences());
        pair_temp.second = mbr.GetBayesRisk();
        one_best_string = ConvertToString(mbr.GetOneBest());

        am_score = pair_temp.first;
        lm_score = pair_temp.second;

        // KALDI_LOG << utt << " 1best Confidence Score:" << pair_temp.first << " BayesRisk:" << pair_temp.second << " result:" << one_best_string;

        // std::cout << "1best Confidence Score: " << pair_temp.first << std::endl;
        // std::cout << "BayesRisk: " << pair_temp.second << std::endl;
        // std::cout << "result: " << one_best_string << std::endl;
    }

    BaseFloat FaiDecoder::GetAvgConf(std::vector<BaseFloat> conf)
    {
        BaseFloat conf_avg = 0.0;

        for (int i = 0; i < conf.size(); i++)
        {
            conf_avg += conf[i];
        }

        conf_avg /= conf.size();

        return conf_avg;
    }

    void FaiDecoder::ApplyLatticeScale(BaseFloat lm_scale, BaseFloat acoustic_scale, CompactLattice &lat)
    {
        std::vector<std::vector<double>> scale(2);
        scale[0].resize(2);
        scale[1].resize(2);
        scale[0][0] = lm_scale;
        scale[0][1] = 0.0;
        scale[1][0] = 0.0;
        scale[1][1] = acoustic_scale;

        ScaleLattice(scale, &lat);
    }

    void FaiDecoder::GetBestPath(const CompactLattice &clat, Lattice &best_path_lat)
    {
        CompactLattice best_path_clat;
        CompactLatticeShortestPath(clat, &best_path_clat);
        ConvertLattice(best_path_clat, &best_path_lat);
    }

    std::string FaiDecoder::ConvertToString(std::vector<std::string> words)
    {
        std::string result_str = "";

        for (int w = 0; w < words.size(); w++)
        {
            result_str += words[w];

            if (w < (words.size() - 1))
            {
                result_str += " ";
            }
        }

        return result_str;
    }

    std::string FaiDecoder::ConvertToString(std::vector<int32> words)
    {
        std::string result_str = "";

        for (int w = 0; w < words.size(); w++)
        {
            result_str += word_syms_->Find(words[w]);

            if (w < (words.size() - 1))
            {
                result_str += " ";
            }
        }

        return result_str;
    }

    void FaiDecoder::GetPrintInfo(OnlineTimer &decoding_timer, const std::string &utt,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like, int32 deb_lv,
                                  std::vector<std::pair<std::string, std::string>> &pair_out,
                                  std::vector<std::pair<std::string, std::string>> &pair_word_align_out,
                                  std::vector<std::pair<std::string, std::string>> &pair_phone_align_out,
                                  bool detokenized_flag)
    {
        if (clat.NumStates() == 0)
        {
            KALDI_WARN << "Empty lattice.";
            return;
        }

        std::vector<int32> words;
        double likelihood;
        LatticeWeight weight;
        int32 num_frames;
        std::vector<int32> alignment;
        std::pair<std::string, std::string> pair_temp;
        Detokenizer tkn;

        CompactLattice best_path_clat;
        Lattice best_path_lat;
        CompactLattice aligned_clat;
        CompactLatticeShortestPath(clat, &best_path_clat);
        ConvertLattice(best_path_clat, &best_path_lat);

        std::string result_str = "";
        std::vector<std::string> words_vec;

        result_str = "";
        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);

        num_frames = alignment.size();
        likelihood = -(weight.Value1() + weight.Value2());
        *tot_num_frames += num_frames;
        *tot_like += likelihood;

        // KALDI_LOG << "Likelihood per frame for utterance " << utt << " " << "1best is "
        //         << (likelihood / num_frames) << " over " << num_frames
        //         << " frames, = " << (-weight.Value1() / num_frames)
        //         << ',' << (weight.Value2() / num_frames);

        if (word_syms_ != NULL)
        {
            for (size_t i = 0; i < words.size(); i++)
            {
                std::string s = word_syms_->Find(words[i]);
                if (s == "")
                    KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
                result_str += s;

                words_vec.push_back(s);

                if (i < (words.size() - 1))
                {
                    result_str += " ";
                }
            }

            if (detokenized_flag)
                result_str = ConvertToString(tkn.detokenizedWords(words_vec));

            pair_temp.first = utt;
            pair_temp.second = result_str;

            // std::cout << "1best: " << utt << " " << result_str << std::endl;
        }

        pair_out.push_back(pair_temp);

        GetWordAlign(best_path_clat, pair_word_align_out, utt);
        GetPhoneAlign(best_path_clat, pair_phone_align_out, utt, alignment);
    }

    void FaiDecoder::GetPrintInfo(const std::string &utt,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like, int32 deb_lv,
                                  std::vector<std::pair<std::string, std::string>> &pair_out,
                                  std::vector<std::pair<std::string, std::string>> &pair_word_align_out,
                                  std::vector<std::pair<std::string, std::string>> &pair_phone_align_out,
                                  bool detokenized_flag)
    {
        if (clat.NumStates() == 0)
        {
            KALDI_WARN << "Empty lattice.";
            return;
        }

        std::vector<int32> words;
        double likelihood;
        LatticeWeight weight;
        int32 num_frames;
        std::vector<int32> alignment;
        std::pair<std::string, std::string> pair_temp;
        Detokenizer tkn;

        CompactLattice best_path_clat;
        Lattice best_path_lat;
        CompactLattice aligned_clat;
        CompactLatticeShortestPath(clat, &best_path_clat);
        ConvertLattice(best_path_clat, &best_path_lat);

        std::string result_str = "";
        std::vector<std::string> words_vec;

        result_str = "";
        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);

        num_frames = alignment.size();
        likelihood = -(weight.Value1() + weight.Value2());
        *tot_num_frames += num_frames;
        *tot_like += likelihood;

        // KALDI_LOG << "Likelihood per frame for utterance " << utt << " " << "1best is "
        //         << (likelihood / num_frames) << " over " << num_frames
        //         << " frames, = " << (-weight.Value1() / num_frames)
        //         << ',' << (weight.Value2() / num_frames);

        if (word_syms_ != NULL)
        {
            for (size_t i = 0; i < words.size(); i++)
            {
                std::string s = word_syms_->Find(words[i]);
                if (s == "")
                    KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
                result_str += s;

                words_vec.push_back(s);

                if (i < (words.size() - 1))
                {
                    result_str += " ";
                }
            }

            if (detokenized_flag)
                result_str = ConvertToString(tkn.detokenizedWords(words_vec));

            pair_temp.first = utt;
            pair_temp.second = result_str;

            // std::cout << "1best: " << utt << " " << result_str << std::endl;
        }

        pair_out.push_back(pair_temp);

        GetWordAlign(best_path_clat, pair_word_align_out, utt);
        GetPhoneAlign(best_path_clat, pair_phone_align_out, utt, alignment);
    }

    void FaiDecoder::GetWordAlign(CompactLattice &best_path_clat, std::vector<std::pair<std::string, std::string>> &pair_align_out, std::string utt)
    {
        bool ok_flag = false;
        int32 max_states = 0;
        bool print_silence = true;

        CompactLattice aligned_clat;
        std::pair<std::string, std::string> pair_align;
        std::vector<int32> words_ali, times, lengths;
        std::stringstream ss;

        ok_flag = WordAlignLattice(best_path_clat, trans_model_, *wordboudary_info_, max_states, &aligned_clat);

        if (!ok_flag)
        {
            KALDI_WARN << "Word Align Error";
            return;
        }
        else
        {
            if (aligned_clat.Start() == fst::kNoStateId)
            {
                KALDI_WARN << "Lattice was empty for key";
                return;
            }
            else
            {
                TopSortCompactLatticeIfNeeded(&aligned_clat);
            }
        }

        ok_flag = CompactLatticeToWordAlignment(aligned_clat, &words_ali, &times, &lengths);

        if (!ok_flag)
        {
            KALDI_WARN << "Format conversion failed for key ";
            return;
        }
        else
        {
            for (size_t i = 0; i < words_ali.size(); i++)
            {
                ss.str("");
                if (words_ali[i] == 0 && !print_silence)
                    continue;

                ss << (frame_shift_ * times[i]) << ' '
                   << (frame_shift_ * lengths[i]) << ' ' << word_syms_->Find(words_ali[i]);

                pair_align.first = utt;
                pair_align.second = ss.str();

                pair_align_out.push_back(pair_align);
            }
        }
    }

    void FaiDecoder::GetPhoneAlign(CompactLattice &best_path_clat, std::vector<std::pair<std::string, std::string>> &pair_align_out, std::string utt, std::vector<int32> &alignment)
    {
        std::stringstream ss;
        std::pair<std::string, std::string> pair_align;
        std::vector<std::vector<int32>> split;
        BaseFloat phone_start = 0.0;
        int32 phone = 0;
        int32 num_repeats = 0;

        SplitToPhones(trans_model_, alignment, &split);

        for (size_t i = 0; i < split.size(); i++)
        {
            KALDI_ASSERT(!split[i].empty());
            ss.str("");

            phone = trans_model_.TransitionIdToPhone(split[i][0]);
            num_repeats = split[i].size();

            std::string p = phone_syms_->Find(phone);

            ss << phone_start << " "
               << (frame_shift_ * num_repeats) << " " << p;

            phone_start += frame_shift_ * num_repeats;
            pair_align.first = utt;
            pair_align.second = ss.str();

            pair_align_out.push_back(pair_align);
        }
    }

    void FaiDecoder::PartialGetPrintInfo(OnlineTimer &decoding_timer, const std::string &utt,
                                         const CompactLattice &clat,
                                         int64 *tot_num_frames,
                                         double *tot_like, int32 deb_lv,
                                         std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag)
    {
        if (clat.NumStates() == 0)
        {
            KALDI_WARN << "Empty lattice.";
            return;
        }

        Lattice best_path_lat;
        double likelihood;
        LatticeWeight weight;
        int32 num_frames;
        std::vector<int32> alignment;
        std::vector<int32> words;
        std::pair<std::string, std::string> pair_temp;
        std::string result_str = "";
        std::vector<std::string> words_vec;
        Lattice decoded;
        CompactLattice decoded_clat;
        Detokenizer tkn;

        CompactLatticeShortestPath(clat, &decoded_clat);
        fst::ConvertLattice(decoded_clat, &decoded);

        if (decoded.Start() == fst::kNoStateId)
            KALDI_ERR << "Failed to get traceback for utterance " << utt;

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
        num_frames = alignment.size();

        likelihood = -(weight.Value1() + weight.Value2());
        *tot_num_frames += num_frames;
        *tot_like += likelihood;

        KALDI_LOG << "Likelihood per frame for utterance " << utt << " " << (1) << "best is "
                  << (likelihood / num_frames) << " over " << num_frames
                  << " frames, = " << (-weight.Value1() / num_frames)
                  << ',' << (weight.Value2() / num_frames);

        if (word_syms_ != NULL)
        {
            for (size_t i = 0; i < words.size(); i++)
            {
                std::string s = word_syms_->Find(words[i]);
                if (s == "")
                    KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
                result_str += s;

                words_vec.push_back(s);

                if (i < (words.size() - 1))
                {
                    result_str += " ";
                }
            }

            if (detokenized_flag)
                result_str = ConvertToString(tkn.detokenizedWords(words_vec));

            pair_temp.first = utt;
            pair_temp.second = result_str;

            std::cout << (1) << "best: " << utt << " " << result_str << std::endl;
        }
        pair_out.push_back(pair_temp);
    }

    string FaiDecoder::PartialGetPrintInfo(const std::string &utt,
                                           const CompactLattice &clat,
                                           int64 *tot_num_frames,
                                           double *tot_like, int32 deb_lv,
                                           std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag)
    {
        if (clat.NumStates() == 0)
        {
            KALDI_WARN << "Empty lattice.";
            return "";
        }

        Lattice best_path_lat;
        double likelihood;
        LatticeWeight weight;
        int32 num_frames;
        std::vector<int32> alignment;
        std::vector<int32> words;
        std::pair<std::string, std::string> pair_temp;
        std::string result_str = "";
        std::vector<std::string> words_vec;
        Lattice decoded;
        CompactLattice decoded_clat;
        Detokenizer tkn;

        CompactLatticeShortestPath(clat, &decoded_clat);
        fst::ConvertLattice(decoded_clat, &decoded);

        if (decoded.Start() == fst::kNoStateId)
            KALDI_ERR << "Failed to get traceback for utterance " << utt;

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
        num_frames = alignment.size();

        likelihood = -(weight.Value1() + weight.Value2());
        *tot_num_frames += num_frames;
        *tot_like += likelihood;

        // KALDI_LOG << "Likelihood per frame for utterance " << utt << " " << (1) <<"best is "
        //                 << (likelihood / num_frames) << " over " << num_frames
        //                 << " frames, = " << (-weight.Value1() / num_frames)
        //                 << ',' << (weight.Value2() / num_frames);

        if (word_syms_ != NULL)
        {
            for (size_t i = 0; i < words.size(); i++)
            {
                std::string s = word_syms_->Find(words[i]);
                if (s == "")
                    KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
                result_str += s;

                words_vec.push_back(s);

                if (i < (words.size() - 1))
                {
                    result_str += " ";
                }
            }

            if (detokenized_flag)
                result_str = ConvertToString(tkn.detokenizedWords(words_vec));

            pair_temp.first = utt;
            pair_temp.second = result_str;

            // std::cout << (1) << "best: " << utt << " " << result_str << std::endl;
        }
        return result_str;
    }

    void FaiDecoder::GetPrintInfo(OnlineTimer &decoding_timer, const std::string &utt,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like, int32 deb_lv, int32 nbest_size,
                                  std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag)
    {
        if (clat.NumStates() == 0)
        {
            KALDI_WARN << "Empty lattice.";
            return;
        }

        Lattice best_path_lat;
        double likelihood;
        LatticeWeight weight;
        int32 num_frames;
        std::vector<int32> alignment;
        std::vector<int32> words;
        std::pair<std::string, std::string> pair_temp;
        std::string result_str = "";
        std::vector<std::string> words_vec;
        std::vector<Lattice> nbest_lats;
        Lattice nbest_lat, lat;
        Detokenizer tkn;

        KALDI_VLOG(4) << "internal timer1:" << decoding_timer.Elapsed();
        ConvertLattice(clat, &lat);
        KALDI_VLOG(4) << "internal timer2:" << decoding_timer.Elapsed();
        fst::ShortestPath(lat, &nbest_lat, nbest_size);
        KALDI_VLOG(4) << "internal timer3:" << decoding_timer.Elapsed();
        fst::ConvertNbestToVector(nbest_lat, &nbest_lats);
        KALDI_VLOG(4) << "internal timer4:" << decoding_timer.Elapsed();
        nbest_size = nbest_lats.size();

        if (nbest_lats.empty())
            KALDI_WARN << "Possibly empty lattice";

        for (int m = 0; m < nbest_lats.size(); m++)
        {
            result_str = "";
            GetLinearSymbolSequence(nbest_lats[m], &alignment, &words, &weight);

            num_frames = alignment.size();
            likelihood = -(weight.Value1() + weight.Value2());
            *tot_num_frames += num_frames;
            *tot_like += likelihood;

            KALDI_LOG << "Likelihood per frame for utterance " << utt << " " << (m + 1) << "best is "
                      << (likelihood / num_frames) << " over " << num_frames
                      << " frames, = " << (-weight.Value1() / num_frames)
                      << ',' << (weight.Value2() / num_frames);

            if (word_syms_ != NULL)
            {
                for (size_t i = 0; i < words.size(); i++)
                {
                    std::string s = word_syms_->Find(words[i]);
                    if (s == "")
                        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
                    result_str += s;

                    words_vec.push_back(s);

                    if (i < (words.size() - 1))
                    {
                        result_str += " ";
                    }
                }

                if (detokenized_flag)
                    result_str = ConvertToString(tkn.detokenizedWords(words_vec));

                pair_temp.first = utt;
                pair_temp.second = result_str;

                std::cout << (m + 1) << "best: " << utt << " " << result_str << std::endl;
            }
            pair_out.push_back(pair_temp);
        }
    }

    void FaiDecoder::GetPrintInfo(const std::string &utt,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like, int32 deb_lv, int32 nbest_size,
                                  std::vector<std::pair<std::string, std::string>> &pair_out, bool detokenized_flag)
    {
        if (clat.NumStates() == 0)
        {
            KALDI_WARN << "Empty lattice.";
            return;
        }

        Lattice best_path_lat;
        double likelihood;
        LatticeWeight weight;
        int32 num_frames;
        std::vector<int32> alignment;
        std::vector<int32> words;
        std::pair<std::string, std::string> pair_temp;
        std::string result_str = "";
        std::vector<std::string> words_vec;
        std::vector<Lattice> nbest_lats;
        Lattice nbest_lat, lat;
        Detokenizer tkn;

        ConvertLattice(clat, &lat);
        fst::ShortestPath(lat, &nbest_lat, nbest_size);
        fst::ConvertNbestToVector(nbest_lat, &nbest_lats);
        nbest_size = nbest_lats.size();

        if (nbest_lats.empty())
            KALDI_WARN << "Possibly empty lattice";

        for (int m = 0; m < nbest_lats.size(); m++)
        {
            result_str = "";
            GetLinearSymbolSequence(nbest_lats[m], &alignment, &words, &weight);

            num_frames = alignment.size();
            likelihood = -(weight.Value1() + weight.Value2());
            *tot_num_frames += num_frames;
            *tot_like += likelihood;

            KALDI_LOG << "Likelihood per frame for utterance " << utt << " " << (m + 1) << "best is "
                      << (likelihood / num_frames) << " over " << num_frames
                      << " frames, = " << (-weight.Value1() / num_frames)
                      << ',' << (weight.Value2() / num_frames);

            if (word_syms_ != NULL)
            {
                for (size_t i = 0; i < words.size(); i++)
                {
                    std::string s = word_syms_->Find(words[i]);
                    if (s == "")
                        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
                    result_str += s;

                    words_vec.push_back(s);

                    if (i < (words.size() - 1))
                    {
                        result_str += " ";
                    }
                }

                if (detokenized_flag)
                    result_str = ConvertToString(tkn.detokenizedWords(words_vec));

                pair_temp.first = utt;
                pair_temp.second = result_str;

                std::cout << (m + 1) << "best: " << utt << " " << result_str << std::endl;
            }
            pair_out.push_back(pair_temp);
        }
    }

    fst::Fst<fst::StdArc> *FaiDecoder::get_fst_addr()
    {
        return decode_fst_;
    }

    fst::SymbolTable *FaiDecoder::get_word_syms_addr()
    {
        return word_syms_;
    }

    OnlineNnet2FeaturePipelineInfo *FaiDecoder::get_fearute_info_addr()
    {
        return feature_info_;
    }

    nnet3::AmNnetSimple *FaiDecoder::get_am_nnet_addr()
    {
        return am_nnet_;
    }

    BaseFloat FaiDecoder::getAmScore()
    {
        return am_score;
    }

    double FaiDecoder::getLmScore()
    {
        return lm_score;
    }

    FaiDecoder::~FaiDecoder()
    {
        delete decoder_;
        delete silence_weighting_;
        delete feature_pipeline_;
        delete cmvn_state_;
        delete adaptation_state_;
        // delete feature_info_;
        // delete word_syms_;

        if (phone_syms_)
        {
            delete phone_syms_;
        }
        // delete decode_fst_;
        delete decodable_info_;

        if (wordboudary_info_)
        {
            delete wordboudary_info_;
        }
    }

}
