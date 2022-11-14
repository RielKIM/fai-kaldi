#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "lat/kaldi-lattice.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include "lat/sausages.h"
#include "detokenizer/detokenizer.h"
#include "lat/word-align-lattice.h"
#include "hmm/hmm-utils.h"

namespace kaldi {
    class FaiDecoder{
        public:
			FaiDecoder();
            FaiDecoder(const OnlineNnet2FeaturePipelineConfig& feature_opts,
                       const nnet3::NnetSimpleLoopedComputationOptions& decodable_opts,
                       const LatticeFasterDecoderConfig& decoder_opts, WordBoundaryInfoNewOpts wordboundary_info_opts,
                       const OnlineEndpointConfig& endpoint_opts, const bool online, const std::string nnet3_rxfilename,
                       const std::string fst_rxfilename, const std::string word_syms_rxfilename, const std::string word_boundary_rxfilename,
                       const std::string phone_syms_rxfilename,
                       const std::string ctm_wxfilename, const BaseFloat chunk_length_secs);
            FaiDecoder(const OnlineNnet2FeaturePipelineConfig& feature_opts,
                       const nnet3::NnetSimpleLoopedComputationOptions& decodable_opts,
                       const LatticeFasterDecoderConfig& decoder_opts, WordBoundaryInfoNewOpts wordboundary_info_opts,
                       const OnlineEndpointConfig& endpoint_opts, const bool online, const std::string nnet3_rxfilename,
                       const std::string fst_rxfilename, const std::string word_syms_rxfilename, const std::string word_boundary_rxfilename,
                       const std::string phone_syms_rxfilename,
                       const std::string ctm_wxfilename, const BaseFloat chunk_length_secs, fst::Fst<fst::StdArc>* fst_addr,
                       fst::SymbolTable* word_syms_addr, OnlineNnet2FeaturePipelineInfo* feature_info_addr, nnet3::AmNnetSimple *am_nnet_addr);

			void SetConfigure(const OnlineNnet2FeaturePipelineConfig& feature_opts,
							const nnet3::NnetSimpleLoopedComputationOptions& decodable_opts,
							const LatticeFasterDecoderConfig& decoder_opts, WordBoundaryInfoNewOpts wordboundary_info_opts, 
							const OnlineEndpointConfig& endpoint_opts, const bool online, const std::string nnet3_rxfilename, 
							const std::string fst_rxfilename, const std::string word_syms_rxfilename, const std::string word_boundary_rxfilename, 
							const std::string phone_syms_rxfilename,
							const std::string ctm_wxfilename, const BaseFloat chunk_length_secs);
			void PruneLatticeW(BaseFloat beam, CompactLattice &prune_clat);
			void GetConfBR(const CompactLattice &clat, std::pair<BaseFloat, double> &pair_temp, const std::string utt);
			BaseFloat GetAvgConf(std::vector<BaseFloat> conf);
            void ApplyLatticeScale(BaseFloat lm_scale, BaseFloat acoustic_scale, CompactLattice &lat);
			void GetBestPath(const CompactLattice &clat, Lattice &best_path_lat);
		
			void PartialGetPrintInfo(OnlineTimer &decoding_timer, const std::string &utt,
                            const CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag);
			string PartialGetPrintInfo(const std::string &utt,
                            const CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag);
			
			void GetPrintInfo(OnlineTimer &decoding_timer, const std::string &utt,
                            const CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv, int32 nbest_size,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag);
			void GetPrintInfo(const std::string &utt,
                            const CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv, int32 nbest_size,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag);

			void GetWordAlign(CompactLattice &best_path_clat, std::vector<std::pair<std::string, std::string> > &pair_align_out, std::string utt);
			void GetPhoneAlign(CompactLattice &best_path_clat, std::vector<std::pair<std::string, std::string> > &pair_align_out, std::string utt, std::vector<int32> &alignment);

			// use word,phone alignment
			void GetPrintInfo(OnlineTimer &decoding_timer, const std::string &utt,
                            const CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv,
                            std::vector<std::pair<std::string, std::string> > &pair_out,
				  			std::vector<std::pair<std::string, std::string> > &pair_word_align_out,
				  			std::vector<std::pair<std::string, std::string> > &pair_phone_align_out,
                            bool detokenized_flag);
			void GetPrintInfo(const std::string &utt,
                            const CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv,
                            std::vector<std::pair<std::string, std::string> > &pair_out,
				  			std::vector<std::pair<std::string, std::string> > &pair_word_align_out,
				  			std::vector<std::pair<std::string, std::string> > &pair_phone_align_out,
                            bool detokenized_flag);

			void SilenceWeightUpdate(std::vector<std::pair<int32, BaseFloat> > delta_weights);
			std::string ConvertToString(std::vector<int32> words);
			std::string ConvertToString(std::vector<std::string> words);
			fst::Fst<fst::StdArc>* get_fst_addr();
			fst::SymbolTable* get_word_syms_addr();
			OnlineNnet2FeaturePipelineInfo* get_fearute_info_addr();
			nnet3::AmNnetSimple* get_am_nnet_addr();
			void AdvanceDecoding();
			bool EndpointDetected();
			void FinalizeDecoding();
			void Init();
			void Init(fst::Fst<fst::StdArc>* fst_addr, fst::SymbolTable* word_syms_addr, OnlineNnet2FeaturePipelineInfo* feature_info_addr, nnet3::AmNnetSimple *am_nnet_addr);
			void Reset();
			void LoadModel();
			void LoadModel(fst::Fst<fst::StdArc>* fst_addr, fst::SymbolTable* word_syms_addr, nnet3::AmNnetSimple *am_nnet_addr);
			void InputFinished();
			void AcceptWaveform(BaseFloat samp_freq, SubVector<BaseFloat> wave_part);
			void GetStateInfo();
			
			void GetRecognitionPartialResult(OnlineTimer &decoding_timer, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                            CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag);

		    string GetRecognitionPartialResult(const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                            CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag);

			void GetRecognitionResult(OnlineTimer &decoding_timer, const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                            CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv, int32 nbest_size,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag, bool confBR_flag);

			void GetRecognitionResult(const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                            CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv, int32 nbest_size,
                            std::vector<std::pair<std::string, std::string> > &pair_out, bool detokenized_flag, bool confBR_flag);

			void GetRecognitionResult(OnlineTimer &decoding_timer, const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                            CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv, int32 nbest_size,
                            std::vector<std::pair<std::string, std::string> > &pair_out, std::vector<std::pair<std::string, std::string> > &pair_word_align_out,
				  			std::vector<std::pair<std::string, std::string> > &pair_phone_align_out, 
				  			bool detokenized_flag, bool confBR_flag);

			void GetRecognitionResult(const bool end_of_utterance, const BaseFloat lm_scale, const BaseFloat wip, const std::string &utt,
                            CompactLattice &clat,
                            int64 *tot_num_frames,
                            double *tot_like, int32 deb_lv, int32 nbest_size,
                            std::vector<std::pair<std::string, std::string> > &pair_out, std::vector<std::pair<std::string, std::string> > &pair_word_align_out,
				  			std::vector<std::pair<std::string, std::string> > &pair_phone_align_out, 
				  			bool detokenized_flag, bool confBR_flag);
			
			BaseFloat getAmScore();
			double getLmScore();

            ~FaiDecoder();

		private:
			OnlineNnet2FeaturePipelineConfig feature_opts_;
			nnet3::NnetSimpleLoopedComputationOptions decodable_opts_;
			LatticeFasterDecoderConfig decoder_opts_;
			OnlineEndpointConfig endpoint_opts_;
			WordBoundaryInfoNewOpts wordboundary_info_opts_;

			Matrix<double> global_cmvn_stats_;
			TransitionModel trans_model_;
		    nnet3::AmNnetSimple *am_nnet_;

			OnlineNnet2FeaturePipelineInfo* feature_info_;
			nnet3::DecodableNnetSimpleLoopedInfo* decodable_info_;
			WordBoundaryInfo* wordboudary_info_;			
			SingleUtteranceNnet3Decoder* decoder_;
			fst::Fst<fst::StdArc> *decode_fst_;
			fst::SymbolTable *word_syms_;
			fst::SymbolTable *phone_syms_;

			OnlineIvectorExtractorAdaptationState* adaptation_state_;
			OnlineCmvnState* cmvn_state_;
			OnlineNnet2FeaturePipeline* feature_pipeline_;
			OnlineSilenceWeighting* silence_weighting_;

			bool online_;
			BaseFloat chunk_length_secs_;
			BaseFloat frame_shift_;
			BaseFloat am_score;
			double lm_score;
			std::string nnet3_rxfilename_;
		    std::string fst_rxfilename_;
			std::string word_syms_rxfilename_;
			std::string word_boundary_rxfilename_;
			std::string phone_syms_rxfilename_;
			std::string ctm_wxfilename_;
    };
}
