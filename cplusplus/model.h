/*
 * model.h
 *
 */

#ifndef MODEL_H_
#define MODEL_H_

#include "corpus.h"

class Model {
private:
	// 一些先验参数和固定配置
	int k_global, k_local;	// global topic数目和local topic数目
	double gamma;
	double alpha_global, beta_global;  // global topic的超参数
	double alpha_local, beta_local;  // local topic的超参数
	double alpha_mix_global, alpha_mix_local;  // loc or gl？ beta分布的参数
	int slidding_window_width; // 滑动窗口宽度
	int vocabulary_size; // 词表大小

	int max_iterator;
	std::string corpus_path; // corpus path
	std::string result_path;	// 模型的result_path
	int save_iter; // for model saving
	std::string save_prefix;	// for model saving

public:
	// 语料库
	Corpus * corpus;

public:
	// 训练过程中需要用到的一些中间数据
	int * global_topic_wc; // n_gl_z
	int * local_topic_wc;  // n_loc_z
	int * global_topic_wc_with_wordid; // n_gl_z_w
	int * local_topic_wc_with_wordid;  // n_loc_z_w

	// model信息；for inference
	int * model_gl_topic_wc;
	int * model_loc_topic_wc;
	int * model_gl_topic_wc_with_wordid;
	int * model_loc_topic_wc_with_wordid;

	double * p;

public:
	Model(int k_gl, int k_loc, double gam, double alpha_gl, double beta_gl,
			double alpha_loc, double beta_loc, double alpha_mix_gl,
			double alpha_mix_loc, int s_window_width, int d_num,
			int v_size, int max_it, std::string cor_path, int save_it,
			std::string save_pre):
			k_global(k_gl), k_local(k_loc), gamma(gam), alpha_global(alpha_gl),
			alpha_local(alpha_loc), beta_global(beta_gl), beta_local(beta_loc),
			alpha_mix_global(alpha_mix_gl), alpha_mix_local(alpha_mix_loc),
			slidding_window_width(s_window_width),
			vocabulary_size(v_size), max_iterator(max_it),
			corpus_path(cor_path), save_iter(save_it), save_prefix(save_pre){
		corpus = new Corpus();
		global_topic_wc = NULL;
		local_topic_wc = NULL;
		global_topic_wc_with_wordid = NULL;
		local_topic_wc_with_wordid = NULL;

		model_gl_topic_wc = NULL;
		model_loc_topic_wc = NULL;
		model_gl_topic_wc_with_wordid = NULL;
		model_loc_topic_wc_with_wordid = NULL;
		p = NULL;
	}
	~Model() {
		if (corpus) {
			delete corpus;
		}
		if (global_topic_wc) {
			delete [] global_topic_wc;
		}
		if (global_topic_wc_with_wordid) {
			delete [] global_topic_wc_with_wordid;
		}
		if (local_topic_wc) {
			delete [] local_topic_wc;
		}
		if (local_topic_wc_with_wordid) {
			delete [] local_topic_wc_with_wordid;
		}
		if (p) {
			delete [] p;
		}
	}
	int Initialize();
	int GibbsSampling();
	int Estimate();
	int SavePhiResult(int iter);
	int SaveTokenResult(int iter);
	int LoadTokenResult();
	int SaveThetaResult(int iter);
	int SaveArgs();
	int Inference();
	int SaveVocabularyResult(int iter);
	int LoadModel(const std::string args_file,
			const std::string vocabulary_file);
	int InfGibbsSampling();
};



#endif /* MODEL_H_ */
