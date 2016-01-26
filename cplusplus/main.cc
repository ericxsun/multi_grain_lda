/*
 * main.cc
 *
 */
#include <cstdio>

#include "model.h"

std::string option = "";

// 参数解析
int parse_args(int argc, char ** argv, int * k_gl, int * k_loc, double * ga,
		double * alpha_gl, double * beta_gl, double * alpha_loc,
		double * beta_loc, double * alpha_mix_gl, double * alpha_mix_loc,
		int * s_win_wid, int * d_num, int * v_size, int * max_it,
		std::string * cor_path, int * save_it, std::string * save_pre,
		std::string * option, std::string * args_file, std::string * voc_file) {
	int i = 1;
	//bool flag_option = false;
    bool flag_k_gl = false, flag_k_local = false,
			flag_alpha_gl = false, flag_alpha_local = false, flag_beta_gl = false,
			flag_beta_loc = false, flag_alpha_mix_gl = false, flag_alpha_mix_loc = false,
			flag_gamma = false, flag_s_win_wid = false,
			flag_v_size = false, flag_corpus_path = false, flag_max_iter = false,
			flag_save_iter = false, flag_save_pre = false, flag_model_args_path = false,
			flag_model_voc_path = false;
	while (i < argc) {
		std::string arg = argv[i];
		if (arg == "--option") {
			//flag_option = true;
			*option = argv[++i];
		} else if (arg == "--k_global") {
			flag_k_gl = true;
			*k_gl = atoi(argv[++i]);
		} else if (arg == "--k_local") {
			flag_k_local = true;
			*k_loc = atoi(argv[++i]);
		} else if (arg == "--gamma") {
			flag_gamma = true;
			*ga = atof(argv[++i]);
		} else if (arg == "--alpha_global") {
			flag_alpha_gl = true;
			*alpha_gl = atof(argv[++i]);
		} else if (arg == "--beta_global") {
			flag_beta_gl = true;
			*beta_gl = atof(argv[++i]);
		} else if (arg == "--alpha_local") {
			flag_alpha_local = true;
			*alpha_loc = atof(argv[++i]);
		} else if (arg == "--beta_local") {
			flag_beta_loc = true;
			*beta_loc = atof(argv[++i]);
		} else if (arg == "--alpha_mix_global") {
			flag_alpha_mix_gl = true;
			*alpha_mix_gl = atof(argv[++i]);
		} else if (arg == "--alpha_mix_local") {
			flag_alpha_mix_loc = true;
			*alpha_mix_loc = atof(argv[++i]);
		} else if (arg == "--slidding_window_width") {
			flag_s_win_wid = true;
			*s_win_wid = atoi(argv[++i]);
		} else if (arg == "--vocabulary_size") {
			flag_v_size = true;
			*v_size = atoi(argv[++i]);
		} else if (arg == "--max_iterator") {
			flag_max_iter = true;
			*max_it = atoi(argv[++i]);
		} else if (arg == "--corpus_path") {
			flag_corpus_path = true;
			*cor_path = argv[++i];
		} else if (arg == "--save_step") {
			flag_save_iter = true;
			*save_it = atoi(argv[++i]);
		} else if (arg == "--save_prefix") {
			flag_save_pre = true;
			*save_pre = argv[++i];
		} else if (arg == "--model_args_file") {
			flag_model_args_path = true;
			*args_file = argv[++i];
		} else if (arg == "--model_vocabulary_file") {
			flag_model_voc_path = true;
			*voc_file = argv[++i];
		}else {
			fprintf(stderr,
					"%s %d ERROR: Invalid args: %s!\n",
					__FILE__,
					__LINE__,
					arg.c_str());
		}
		++i;
	}
	if (*option == "load_corpus") {
		if (!(flag_alpha_gl && flag_beta_gl && flag_gamma && flag_alpha_mix_gl
				&& flag_alpha_local && flag_beta_loc && flag_alpha_mix_loc
				&& flag_k_gl && flag_k_local && flag_s_win_wid
				&& flag_max_iter && flag_save_iter && flag_save_pre && flag_corpus_path
				&& flag_v_size)) {
			fprintf(stderr,
					"%s %d ERROR: Option load_corpus has some args lost!\n",
					__FILE__,
					__LINE__);
			return -1;
		}
	} else if (*option == "load_result") {
		if (!(flag_alpha_gl && flag_beta_gl && flag_gamma && flag_alpha_mix_gl
				&& flag_alpha_local && flag_beta_loc && flag_alpha_mix_loc
				&& flag_k_gl && flag_k_local && flag_s_win_wid
				&& flag_max_iter && flag_save_iter && flag_save_pre && flag_corpus_path
				&& flag_v_size)) {
			fprintf(stderr,
					"%s %d ERROR: Option load_result has some args lost!\n",
					__FILE__,
					__LINE__);
			return -1;
		}
	} else if (*option == "inference") {
		if (!(flag_corpus_path && flag_max_iter && flag_save_iter
				&& flag_model_args_path && flag_model_voc_path)) {
			fprintf(stderr,
					"%s %d ERROR: Option inference has some args lost!\n",
					__FILE__,
					__LINE__);
			return -1;
		}
	} else {
		fprintf(stderr,
				"%s %d ERROR: Invalid option: %s\n",
				__FILE__,
				__LINE__,
				option->c_str());
		return -1;
	}
	return 0;
}

int main(int argc, char ** argv) {
	int k_gl, k_loc;
	double ga, alpha_gl, beta_gl, alpha_loc, beta_loc, alpha_mix_gl,
			alpha_mix_loc;
	int s_win_wid, d_num, v_size, max_it;
	std::string cor_path;
	int save_it;
	std::string save_pre;
	std::string args_file;
	std::string voc_file;
	int ret = parse_args(argc, argv, &k_gl, &k_loc, &ga, &alpha_gl, &beta_gl,
			&alpha_loc, &beta_loc, &alpha_mix_gl, &alpha_mix_loc, &s_win_wid,
			&d_num, &v_size, &max_it, &cor_path, &save_it, &save_pre, &option,
			&args_file, &voc_file);
	if (ret != 0) {
		fprintf(stderr,
				"%s %d ERROR: Something error when parsing args!\n",
				__FILE__,
				__LINE__);
		return -1;
	}

	if (option == "load_corpus") {
		Model model(k_gl, k_loc, ga, alpha_gl, beta_gl, alpha_loc, beta_loc,
				alpha_mix_gl, alpha_mix_loc, s_win_wid, d_num, v_size, max_it,
				cor_path, save_it, save_pre);
		model.SaveArgs();
		if (model.Initialize() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Model initialize error\n",
					__FILE__,
					__LINE__);
			return -1;
		}
		if (model.Estimate() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Model estimate error!\n",
					__FILE__,
					__LINE__);
			return -1;
		}
		fprintf(stderr,
				"%s %d INFO: Model Estimate success!\n",
				__FILE__,
				__LINE__);
		return 0;
	} else if (option == "load_result") {
		Model model(k_gl, k_loc, ga, alpha_gl, beta_gl, alpha_loc, beta_loc,
				alpha_mix_gl, alpha_mix_loc, s_win_wid, d_num, v_size, max_it,
				cor_path, save_it, save_pre);
		model.SaveArgs();
		if (model.LoadTokenResult() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Load token result error\n",
					__FILE__,
					__LINE__);
			return -1;
		}
		if (model.Estimate() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Model estimate error!\n",
					__FILE__,
					__LINE__);
			return -1;
		}
		fprintf(stderr,
				"%s %d INFO: Model estimate success!\n",
				__FILE__,
				__LINE__);
		return 0;
	} else if (option == "inference") {
		Model model(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 0, d_num, 0, max_it,
				cor_path, save_it, save_pre);
		model.LoadModel(args_file, voc_file);
		if (model.Initialize() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Model initialize error!\n",
					__FILE__,
					__LINE__);
			return -1;
		}
		if (model.Inference() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Inference error!\n",
					__FILE__,
					__LINE__);
			return -1;
		}
		fprintf(stderr,
				"%s %d INFO: Inference success!\n",
				__FILE__,
				__LINE__);
		return 0;
	} else {
		fprintf(stderr,
				"%s %d ERROR: Invalid option: %s",
				__FILE__,
				__LINE__,
				option.c_str());
		return -1;
	}
}
