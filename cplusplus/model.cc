/*
 * model.cc
 *
 */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <assert.h>

#include "strtools.h"
#include "corpus.h"
#include "model.h"

const int MAX_WORD_INFO_LENGTH = 1024;
bool is_debug = false;

int Model::LoadModel(const std::string args_path,
		const std::string vocabulary_path) {
	FILE * args_file = fopen(args_path.c_str(), "r");
	if (!args_file) {
		fprintf(stderr,
				"%s %d ERROR: Open args file error!\n",
				__FILE__,
				__LINE__);
		return -1;
	}
	char buffer[1024];
	bool flag_alpha_gl = false, flag_alpha_loc = false, flag_beta_gl = false,
			flag_beta_loc = false, flag_alpha_mix_gl = false, flag_alpha_mix_loc = false,
			flag_gamma = false, flag_vocabulary_size = false, flag_k_gl = false,
			flag_k_loc = false;
	while (fgets(buffer, 1024, args_file) != NULL) {
		std::string str_buf(buffer);
		std::vector<std::string> items;
		if (StrSplit(str_buf, ':', &items) != 2) {
			memset(buffer, 0, sizeof(char) * 1024);
			continue;
		}
		if (items[0] == "k_global") {
			k_global = atoi(items[1].c_str());
			flag_k_gl = true;
		} else if (items[0] == "k_local") {
			k_local = atoi(items[1].c_str());
			flag_k_loc = true;
		} else if (items[0] == "alpha_global") {
			alpha_global = atof(items[1].c_str());
			flag_alpha_gl = true;
		} else if (items[0] == "alpha_local") {
			alpha_local = atof(items[1].c_str());
			flag_alpha_loc = true;
		} else if (items[0] == "alpha_mix_global") {
			alpha_mix_global = atof(items[1].c_str());
			flag_alpha_mix_gl = true;
		} else if (items[0] == "alpha_mix_local") {
			alpha_mix_local = atof(items[1].c_str());
			flag_alpha_mix_loc = true;
		} else if (items[0] == "beta_global") {
			beta_global = atof(items[1].c_str());
			flag_beta_gl = true;
		} else if (items[0] == "beta_local") {
			beta_local = atof(items[1].c_str());
			flag_beta_loc = true;
		} else if (items[0] == "gamma") {
			gamma = atof(items[1].c_str());
			flag_gamma = true;
		} else if (items[0] == "vocabulary_size") {
			vocabulary_size = atoi(items[1].c_str());
			flag_vocabulary_size = true;
		}
		memset(buffer, 0, sizeof(char) * 1024);
	}
	fclose(args_file);
	if (!(flag_alpha_gl && flag_alpha_loc && flag_alpha_mix_gl
			&& flag_alpha_mix_loc && flag_beta_gl && flag_beta_loc && flag_gamma
			&& flag_k_gl && flag_k_loc && flag_vocabulary_size)) {
		fprintf(stderr,
				"%s %d ERROR: Load args file error!\n",
				__FILE__,
				__LINE__);
		return -1;
	}
	FILE * vocabulary_file = fopen(vocabulary_path.c_str(), "r");
	if (!vocabulary_file) {
		fprintf(stderr,
				"%s %d ERROR: Open vocabulary file error!\n",
				__FILE__,
				__LINE__);
		return -1;
	}
	int global_topic_num;
	if (fscanf(vocabulary_file, "global_topic_num:%d", &global_topic_num) != 1
			|| global_topic_num != k_global) {
		fprintf(stderr,
				"%s %d ERROR: Read vocabulary file error!\n",
				__FILE__,
				__LINE__);
		fclose(vocabulary_file);
		return -1;
	}
	for (int gl_id = 0; gl_id < k_global; ++ gl_id) {
		int cur_id, word_num;
		if (fscanf(vocabulary_file, "gl_topic_%d:%d", &cur_id, &word_num) != 2
				|| cur_id != gl_id) {
			fprintf(stderr,
					"%s %d ERROR: Read vocabulary file error!\n",
					__FILE__,
					__LINE__);
			fclose(vocabulary_file);
			return -1;
		}
		model_gl_topic_wc[gl_id] = word_num;
	}
	int local_topic_num;
	if (fscanf(vocabulary_file, "local_topic_num:%d", &local_topic_num) != 1
			|| local_topic_num != k_local) {
		fprintf(stderr,
				"%s %d ERROR: Read vocabulary file error!\n",
				__FILE__,
				__LINE__);
		fclose(vocabulary_file);
		return -1;
	}
	for (int loc_id = 0; loc_id < k_local; ++ loc_id) {
		int cur_id, word_num;
		if (fscanf(vocabulary_file, "loc_topic_%d:%d", &cur_id, &word_num) != 2
				|| cur_id != loc_id) {
			fprintf(stderr,
					"%s %d ERROR: Read vocabulary file error!\n",
					__FILE__,
					__LINE__);
			fclose(vocabulary_file);
			return -1;
		}
		model_loc_topic_wc[loc_id] = word_num;
	}
	for (int gl_id = 0; gl_id < k_global; ++ gl_id) {
		int cur_global_id;
		if (fscanf(vocabulary_file, "global_topic_id:%d", &cur_global_id) != 1
				|| cur_global_id != gl_id) {
			fprintf(stderr,
					"%s %d ERROR: Read vocabulary file error!\n",
					__FILE__,
					__LINE__);
			fclose(vocabulary_file);
			return -1;
		}
		for (int w_id = 0; w_id < vocabulary_size; ++ w_id) {
			int cur_w_id, word_num;
			if (fscanf(vocabulary_file, "%d:%d", &cur_w_id, &word_num) != 2
					|| cur_w_id != w_id) {
				fprintf(stderr,
						"%s %d ERROR: Read vocabulary file error!\n",
						__FILE__,
						__LINE__);
				fclose(vocabulary_file);
				return -1;
			}
			model_gl_topic_wc_with_wordid[gl_id * vocabulary_size + w_id] = word_num;
		}
	}
	for (int loc_id = 0; loc_id < k_local; ++ loc_id) {
		int cur_local_id;
		if (fscanf(vocabulary_file, "local_topic_id:%d", &cur_local_id) != 1
				|| cur_local_id != loc_id) {
			fprintf(stderr,
					"%s %d ERROR: Read vocabulary file error!\n",
					__FILE__,
					__LINE__);
			fclose(vocabulary_file);
			return -1;
		}
		for (int w_id = 0; w_id < vocabulary_size; ++ w_id) {
			int cur_w_id, word_num;
			if (fscanf(vocabulary_file, "%d:%d", &cur_w_id, &word_num) != 2
					|| cur_w_id != w_id) {
				fprintf(stderr,
						"%s %d ERROR: Read vocabulary file error!\n",
						__FILE__,
						__LINE__);
				fclose(vocabulary_file);
				return -1;
			}
			model_loc_topic_wc_with_wordid[loc_id * vocabulary_size + w_id] = word_num;
		}
	}
	fclose(vocabulary_file);
	return 0;
}

int Model::LoadTokenResult() {
	// 从corpus_path的文件中读取训练结果
	FILE * result_file = fopen(corpus_path.c_str(), "r");
	if (!result_file) {
		fprintf(stderr,
				"%s:%d ERROR: Cannot open result file!\n",
				__FILE__,
				__LINE__);
		return -1;
	}

	int read_doc_num = 0;
	int length, sent_num;
	int doc_id;

	// model级别用于采样的数据结构
	// 设置n_z, n_z_w, p_z等
	if (global_topic_wc
			|| global_topic_wc_with_wordid
			|| local_topic_wc
			|| local_topic_wc_with_wordid
			|| p) {
		fprintf(stderr, "%s:%d ERROR: Init fail!\n", __FILE__, __LINE__);
		fclose(result_file);
		return -1;
	}
	global_topic_wc = new int[k_global];
	memset(global_topic_wc, 0, sizeof(int) * k_global);

	global_topic_wc_with_wordid = new int[k_global * vocabulary_size];
	memset(global_topic_wc_with_wordid,
			0,
			sizeof(int) * k_global * vocabulary_size);

	local_topic_wc = new int[k_local];
	memset(local_topic_wc, 0, sizeof(int) * k_local);

	local_topic_wc_with_wordid = new int[k_local * vocabulary_size];
	memset(local_topic_wc_with_wordid,
			0,
			sizeof(int) * k_local * vocabulary_size);

	p = new double[(k_global + k_local) * slidding_window_width];
	for (int x = 0; x < (k_global + k_local) * slidding_window_width; ++x) {
		p[x] = 0.0;
	}

	while (fscanf(result_file,
			"doc_id:%d sent_num:%d word_num:%d",
			&doc_id,
			&sent_num,
			&length) == 3) {
        DocumentPtr cur_doc = DocumentPtr(new Document());
        corpus->docs.push_back(cur_doc);
        cur_doc->doc_id = doc_id;

		// 文档整体滑动窗口的个数
		cur_doc->sliding_window_num =
				cur_doc->sentences.size() + slidding_window_width - 1;

		if (cur_doc->gl_wc_in_s_window.size()
				|| cur_doc->loc_wc_in_s_window.size()
				|| cur_doc->wc_in_s_window.size()
				|| cur_doc->wc_in_s_window_with_gl_topic.size()
				|| cur_doc->wc_in_s_window_with_loc_topic.size()
				|| cur_doc->wc_with_gl_topic.size()
				|| cur_doc->wc_with_loc_topic.size()) {
			fprintf(stderr,
					"%s:%d ERROR:Init fail! Doc %d's word count pointer is not NULL",
					__FILE__,
					__LINE__,
					read_doc_num);
			fclose(result_file);
			return -1;
		}
		//为滑动窗口相关的抽样统计数据开辟空间并且全都置0
        cur_doc->ResetData(k_global, k_local);

		int read_sent_num = 0;	// 已经读取的句子数目
		int read_word_num = 0;  // 计算offset
		int sent_length;
		int sent_id;

		// 逐行读取句子
		for (; read_sent_num < sent_num; read_sent_num ++) {
			if (fscanf(result_file,
					"sent_id:%d length:%d",
					&sent_id,
					&sent_length) != 2) {
				fprintf(stderr,
						"%s %d ERROR: Read corpus file failed!\n",
						__FILE__,
						__LINE__);
				fclose(result_file);
				return -1;
			}
            SentencePtr cur_sent = SentencePtr(new Sentence());
            cur_sent->sent_id = sent_id;
            cur_sent->offset = read_word_num;
            cur_doc->sentences.push_back(cur_sent);

			// sentence级别用于采样的数据结构
            for(int i=0; i<slidding_window_width; i++){
                cur_sent->wc_which_window.push_back(0);
            }

			int word_id, window_id, gl_topic_id, loc_topic_id;
			// 读取句子中的词
			for (int word_ind = 0; word_ind < sent_length; ++ word_ind) {
				if (fscanf(result_file,
						"%d:%d,%d,%d",
						&word_id,
						&window_id,
						&gl_topic_id,
						&loc_topic_id) != 4) {
					fprintf(stderr,
							"%s %d ERROR: Read corpus file failed!\n",
							__FILE__,
							__LINE__);
					fclose(result_file);
					return -1;
				}
				assert(word_id >= 0 && word_id < vocabulary_size);
				cur_sent->words.push_back(word_id);
				cur_sent->words_global_topic.push_back(gl_topic_id);
				cur_sent->words_local_topic.push_back(loc_topic_id);
				cur_sent->words_which_window.push_back(window_id);
				cur_sent->wc_which_window[window_id] ++;

				// doc级别的统计信息
 				cur_doc->wc_in_s_window[read_sent_num + window_id]++;
				if (gl_topic_id != -1) {
					assert(gl_topic_id >= 0 && gl_topic_id < k_local);
					cur_doc->gl_wc_in_s_window[read_sent_num + window_id]++;
					cur_doc->wc_in_s_window_with_gl_topic[(read_sent_num + window_id) * k_global + gl_topic_id]++;
					cur_doc->wc_with_gl_topic[gl_topic_id]++;
					cur_doc->wc_global++;

					// model级别的统计信息
					global_topic_wc[gl_topic_id]++;
					global_topic_wc_with_wordid[gl_topic_id * vocabulary_size + word_id]++;
				} else if (loc_topic_id != -1) {
					assert(loc_topic_id >= 0 && loc_topic_id < k_local);
					cur_doc->loc_wc_in_s_window[read_sent_num + window_id]++;
					cur_doc->wc_in_s_window_with_loc_topic[(read_sent_num + window_id) * k_local + loc_topic_id]++;
					cur_doc->wc_with_loc_topic[loc_topic_id]++;
					cur_doc->wc_local++;
					//model级别的统计信息
					local_topic_wc[loc_topic_id]++;
					local_topic_wc_with_wordid[loc_topic_id * vocabulary_size + word_id]++;
				} else {
					fprintf(stderr,
							"%s %d ERROR: A word has no topic allocated!\n",
							__FILE__,
							__LINE__);
					fclose(result_file);
					return -1;
				}
			}
			read_word_num += sent_length;
		}
		read_doc_num++;
	}
	return read_doc_num;
}

int Model::Initialize() {
	// 开辟内存空间；置0等
	srand(time(NULL));
	fprintf(stderr,
			"%s %d INFO: Start initializing...\n",
			__FILE__,
			__LINE__);
	fprintf(stderr,
			"%s %d INFO: Start loading corpus...\n",
			__FILE__,
			__LINE__);
	int read_doc_num;
	read_doc_num = corpus->Load(corpus_path);
	fprintf(stderr,
				"%s %d Read corpus ! Have read %d docs\n",
				__FILE__,
				__LINE__,
				read_doc_num);
    if(read_doc_num < 1){
		return -1;
	}
	fprintf(stderr, "%s %d INFO: Load corpus end\n", __FILE__, __LINE__);
	fprintf(stderr,
			"%s %d INFO: Create data struct for Gibbs Sampling...\n",
			__FILE__,
			__LINE__);

	// 设置n_z, n_z_w, p_z等
	if (global_topic_wc
			|| global_topic_wc_with_wordid
			|| local_topic_wc
			|| local_topic_wc_with_wordid
			|| p) {
		fprintf(stderr, "%s %d ERROR: Init fail!\n", __FILE__, __LINE__);
		return -1;
	}

	global_topic_wc = new int[k_global];
	memset(global_topic_wc, 0, sizeof(int) * k_global);

	global_topic_wc_with_wordid = new int[k_global * vocabulary_size];
	memset(global_topic_wc_with_wordid,
			0,
			sizeof(int) * k_global * vocabulary_size);

	local_topic_wc = new int[k_local];
	memset(local_topic_wc, 0, sizeof(int) * k_local);

	local_topic_wc_with_wordid = new int[k_local * vocabulary_size];
	memset(local_topic_wc_with_wordid,
			0,
			sizeof(int) * k_local * vocabulary_size);

	p = new double[(k_global + k_local) * slidding_window_width];
	for (int x = 0; x < (k_global + k_local) * slidding_window_width; ++x) {
		p[x] = 0.0;
	}

	// 针对每个doc的抽样统计数据
	for (uint32_t doc_ind = 0; doc_ind < corpus->docs.size(); ++ doc_ind) {
		DocumentPtr cur_doc = corpus->docs[doc_ind];
		// 滑动窗口数量
		cur_doc->sliding_window_num =
				cur_doc->sentences.size() + slidding_window_width - 1;
		if (cur_doc->gl_wc_in_s_window.size()
				|| cur_doc->loc_wc_in_s_window.size()
				|| cur_doc->wc_in_s_window.size()
				|| cur_doc->wc_in_s_window_with_gl_topic.size()
				|| cur_doc->wc_in_s_window_with_loc_topic.size()
				|| cur_doc->wc_with_gl_topic.size()
				|| cur_doc->wc_with_loc_topic.size()) {
			fprintf(stderr,
					"%s %d ERROR: Init fail! Doc %d's word count pointer is not NULL",
					__FILE__,
					__LINE__,
					doc_ind);
			return -1;
		}
		//为滑动窗口相关的抽样统计数据开辟空间并且全都置0
        cur_doc->ResetData(k_global, k_local);

		// 句子相关的抽样统计数据
		for (uint32_t sent_ind = 0; sent_ind < cur_doc->sentences.size(); ++ sent_ind) {
			SentencePtr cur_sentence = cur_doc->sentences[sent_ind];
			int cur_sentence_length = cur_sentence->words.size();
			if (cur_sentence->words_global_topic.size()
					|| cur_sentence->words_local_topic.size()
					|| cur_sentence->words_which_window.size()
					|| cur_sentence->wc_which_window.size()) {
				fprintf(stderr,
						"%s %d ERROR: Init fail! Doc %d's sentence %d's word's"
						" topic pointer is not NULL!\n",
						__FILE__,
						__LINE__,
						doc_ind,
						sent_ind);
				return -1;
			}

            cur_sentence->ResetData(slidding_window_width);

			// for each words
			for (int word_ind = 0; word_ind < cur_sentence_length; ++ word_ind) {
				int cur_word = cur_sentence->words[word_ind];
                std::cout << "doc_id:" << cur_doc->doc_id << ", sent_id:" << cur_sentence->sent_id << ", wd:" << cur_word << ", vocabulary_size:" << vocabulary_size << std::endl;
				assert(cur_word >= 0 && cur_word < vocabulary_size);
				int window_id = (int) (rand() % slidding_window_width);
				cur_sentence->words_which_window[word_ind] = window_id;
				cur_sentence->wc_which_window[window_id]++;
				cur_doc->wc_in_s_window[sent_ind + window_id]++;
				int is_gl = rand() % 2;	// gl or loc ??
				if (is_gl == 1) { // global_topic
					// random采样，等概率
					int topic = (int) ((double) rand() / RAND_MAX * k_global);
					cur_sentence->words_global_topic[word_ind] = topic;
					cur_sentence->words_local_topic[word_ind] = -1;

					cur_doc->gl_wc_in_s_window[sent_ind + window_id]++;
					cur_doc->wc_in_s_window_with_gl_topic[(sent_ind + window_id) * k_global + topic]++;
					cur_doc->wc_with_gl_topic[topic]++;
					// n_d_gl
					cur_doc->wc_global++;

					global_topic_wc[topic] += 1;
					global_topic_wc_with_wordid[topic * vocabulary_size + cur_word]++;
				} else { // local_topic
					// random采样，等概率
					int topic = (int) ((double) rand() / RAND_MAX * k_local);
					// word对应的topic
					cur_sentence->words_local_topic[word_ind] = topic;
					cur_sentence->words_global_topic[word_ind] = -1;

					cur_doc->loc_wc_in_s_window[sent_ind + window_id]++;
					cur_doc->wc_in_s_window_with_loc_topic[(sent_ind + window_id) * k_local + topic]++;
					cur_doc->wc_with_loc_topic[topic]++;
					cur_doc->wc_local++;

					local_topic_wc[topic]++;
					local_topic_wc_with_wordid[topic * vocabulary_size + cur_word]++;
				}
			}
		}
	}
	fprintf(stderr, "Create data struct for Gibbs Sampling end");
	return 0;
}

int Model::GibbsSampling() {
	srand(time(NULL));

	// 一些全局的double；放在循环体外节省时间
	double v_beta_global = vocabulary_size * beta_global;
	double k_alpha_global = k_global * alpha_global;
	double s_gamma = slidding_window_width * gamma;
	double v_beta_local = vocabulary_size * beta_local;
	double k_alpha_local = k_local * alpha_local;

	// doc_count
	for (uint32_t doc_ind = 0; doc_ind < corpus->docs.size(); ++ doc_ind) {
		// doc_count
		if (doc_ind % 100000 == 0 && doc_ind != 0 && is_debug) {
			fprintf(stderr,
					"%s %d INFO: 100, 000 docs be sampled!\n",
					__FILE__,
					__LINE__);
		}
		DocumentPtr cur_doc = corpus->docs[doc_ind];
		for (uint32_t sent_ind = 0; sent_ind < cur_doc->sentences.size(); ++ sent_ind) {
			SentencePtr cur_sentence = cur_doc->sentences[sent_ind];
			// for each words
			for (uint32_t w_ind = 0; w_ind < cur_sentence->words.size(); ++ w_ind) {
				int cur_word_id = cur_sentence->words[w_ind];
				// which window
				int old_v = cur_sentence->words_which_window[w_ind];
				// which topic
				int old_global_topic = cur_sentence->words_global_topic[w_ind];
				int old_local_topic = cur_sentence->words_local_topic[w_ind];
				// just for exception
				if (old_global_topic >= 0 && old_local_topic >= 0) {
					fprintf(stderr,
							"%s %d ERROR: A word has two topics!\n",
							__FILE__,
							__LINE__);
					return -1;
				}
				// decrease
				if (old_global_topic >= 0) {
					// global_topic
					cur_sentence->words_global_topic[w_ind] = -1;
					cur_sentence->words_which_window[w_ind] = -1;
					cur_sentence->wc_which_window[old_v]--;
					cur_doc->gl_wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window_with_gl_topic[(sent_ind + old_v) * k_global + old_global_topic]--;
					cur_doc->wc_with_gl_topic[old_global_topic]--;
					cur_doc->wc_global--;
					global_topic_wc[old_global_topic]--;
					global_topic_wc_with_wordid[old_global_topic * vocabulary_size + cur_word_id]--;
				} else if (old_local_topic >= 0) {
					// local_topic
					cur_sentence->words_local_topic[w_ind] = -1;
					cur_sentence->words_which_window[w_ind] = -1;
					cur_sentence->wc_which_window[old_v]--;
					cur_doc->loc_wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window_with_loc_topic[(sent_ind + old_v) * k_local + old_local_topic]--;
					cur_doc->wc_with_loc_topic[old_local_topic]--;
					cur_doc->wc_local--;
					local_topic_wc[old_local_topic]--;
					local_topic_wc_with_wordid[old_local_topic * vocabulary_size + cur_word_id]--;
				} else {
					fprintf(stderr, "A words has no topics!\n");
					return -1;
				}

				// sampling
				int sample_iter = 0;
				for (int win_id = 0; win_id < slidding_window_width; ++ win_id) {
					for (int gl_id = 0; gl_id < k_global; ++ gl_id) {
						// global_topic
						// l是slidding window id， m是topic id
						double score = 1.0;
						score = score *
								(double(global_topic_wc_with_wordid[gl_id * vocabulary_size + cur_word_id]) + beta_global)
								/ (double(global_topic_wc[gl_id]) + v_beta_global);
						score = score
								* (double(cur_sentence->wc_which_window[win_id]) + gamma)
								/ (double(cur_sentence->words.size() + s_gamma));
						score = score
								* (double(cur_doc->gl_wc_in_s_window[sent_ind + win_id]) + alpha_mix_global)
								/ (double(cur_doc->wc_in_s_window[sent_ind + win_id]) + alpha_mix_global + alpha_mix_local);
						score = score
								* (double(cur_doc->wc_with_gl_topic[gl_id]) + alpha_global)
								/ (double(cur_doc->wc_global) + k_alpha_global);
						p[sample_iter++] = score;
					}
					for (int loc_id = 0; loc_id < k_local; ++ loc_id) {
						// local_topic
						double score = 1.0;
						score = score *
								((double(local_topic_wc_with_wordid[loc_id * vocabulary_size + cur_word_id])) + beta_local)
								/ (double(local_topic_wc[loc_id]) + v_beta_local);
						score = score *
								(double(cur_sentence->wc_which_window[win_id]) + gamma)
								/ (double(cur_sentence->words.size() + s_gamma));
						score = score *
								(double(cur_doc->loc_wc_in_s_window[sent_ind + win_id]) + alpha_mix_local)
								/ (double(cur_doc->wc_in_s_window[sent_ind + win_id]) + alpha_mix_global + alpha_mix_local);
						score = score
								* (double(cur_doc->wc_in_s_window_with_loc_topic[(sent_ind + win_id) * k_local + loc_id]) + alpha_local)
								/ (double(cur_doc->loc_wc_in_s_window[sent_ind + win_id]) + k_alpha_local);
						p[sample_iter++] = score;
					}
				}

				// multinomial采样
				int p_length = (k_global + k_local) * slidding_window_width;
				for (int z = 1; z < p_length; ++z) {
					p[z] = p[z] + p[z - 1];
				}
				double random_p =
						((double) rand() / RAND_MAX) * p[p_length - 1];
				int new_index = 0;
				for (int z = 0; z < p_length; ++z) {
					if (p[z] >= random_p) {
						new_index = z;
						break;
					}
				}

				// 采样完毕, 计算新采样的词频率
				int new_window = new_index / (k_global + k_local);
				int new_topic_index = new_index % (k_global + k_local);
				int new_global_topic = -1;
				int new_local_topic = -1;
				if (new_topic_index < k_global) {
					new_global_topic = new_topic_index;
				} else {
					new_local_topic = new_topic_index - k_global;
				}

				// increase
				cur_sentence->wc_which_window[new_window]++;
				cur_sentence->words_which_window[w_ind] = new_window;
				cur_sentence->words_global_topic[w_ind] = new_global_topic;
				cur_sentence->words_local_topic[w_ind] = new_local_topic;
				if (new_global_topic >= 0) {
					// global_topic
					cur_doc->wc_in_s_window[sent_ind + new_window]++;
					cur_doc->gl_wc_in_s_window[sent_ind + new_window]++;
					cur_doc->wc_in_s_window_with_gl_topic[(sent_ind + new_window) * k_global	+ new_global_topic]++;
					cur_doc->wc_with_gl_topic[new_global_topic]++;
					cur_doc->wc_global++;
					global_topic_wc[new_global_topic]++;
					global_topic_wc_with_wordid[new_global_topic * vocabulary_size + cur_word_id]++;
				} else if (new_local_topic >= 0) {
					// local_topic
					cur_doc->wc_in_s_window[sent_ind + new_window]++;
					cur_doc->loc_wc_in_s_window[sent_ind + new_window]++;
					cur_doc->wc_in_s_window_with_loc_topic[(sent_ind + new_window) * k_local + new_local_topic]++;
					cur_doc->wc_with_loc_topic[new_local_topic]++;
					cur_doc->wc_local++;
					local_topic_wc[new_local_topic]++;
					local_topic_wc_with_wordid[new_local_topic * vocabulary_size + cur_word_id]++;
				} else {
					fprintf(stderr,
							"%s %d ERROR: A word has no new topic!\n",
							__FILE__,
							__LINE__);
					return -1;
				}
			}
		}
	}
	return 0;
}

// 训练/保存的过程
int Model::Estimate() {
	int cur_iter = 0;
	for (; cur_iter < max_iterator; ++cur_iter) {
		if (GibbsSampling() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Gibbs Sampling iter %d failed!\n",
					__FILE__,
					__LINE__,
					cur_iter);
			return -1;
		}
		fprintf(stderr,
				"%s %d INFO: Gibbs Sampling iter %d success!\n",
				__FILE__,
				__LINE__,
				cur_iter);
		if ((cur_iter + 1) % (save_iter * 5) == 0) {
			SaveVocabularyResult(cur_iter);
			SavePhiResult(cur_iter);
			SaveThetaResult(cur_iter);
			SaveTokenResult(cur_iter);
		}
		else if ((cur_iter + 1) % save_iter == 0) {
			SavePhiResult(cur_iter);
		}
	}
    std::cout << "SaveVocabularyResult" << std::endl;
	SaveVocabularyResult(cur_iter);
    std::cout << "SavePhiResult" << std::endl;
	SavePhiResult(cur_iter);
    std::cout << "SaveThetaResult" << std::endl;
	SaveThetaResult(cur_iter);
    std::cout << "SaveTokenResult" << std::endl;
	SaveTokenResult(cur_iter);
    std::cout << "end SaveTokenResult" << std::endl;
	return 0;
}

// 保存args
int Model::SaveArgs() {
	char file_name[128];
	snprintf(file_name, 128, "data/%s.mglda_args", save_prefix.c_str());
	FILE * args_file = fopen(file_name, "w");
	fprintf(args_file, "k_global:%d\n", k_global);
	fprintf(args_file, "k_local:%d\n", k_local);
	fprintf(args_file, "alpha_global:%.9f\n", alpha_global);
	fprintf(args_file, "alpha_local:%.9f\n", alpha_local);
	fprintf(args_file, "alpha_mix_global:%.9f\n", alpha_mix_global);
	fprintf(args_file, "alpha_mix_local:%.9f\n", alpha_mix_local);
	fprintf(args_file, "beta_global:%.9f\n", beta_global);
	fprintf(args_file, "beta_local:%.9f\n", beta_local);
	fprintf(args_file, "gamma:%.9f\n", gamma);
	fprintf(args_file, "doc_num:%lu\n", corpus->docs.size());
	fprintf(args_file, "vocabulary_size:%d\n", vocabulary_size);
	fclose(args_file);
	return 0;
}

// 保存topic->word的分布
int Model::SavePhiResult(int iter) {
	char file_name[128];
	snprintf(file_name, 128, "data/%s_local_phi.iter%d", save_prefix.c_str(), iter);
	FILE * loc_phi = fopen(file_name, "w");
	double cur_phi;
	for (int i = 0; i < k_local; ++i) {
		// 计算local_topic的phi
		fprintf(loc_phi, "local_topic_id:%d", i);
		for (int j = 0; j < vocabulary_size; ++j) {
			cur_phi = (double(local_topic_wc_with_wordid[i * vocabulary_size + j]) + beta_local)
					/ (double(local_topic_wc[i]) + vocabulary_size * beta_local);
			fprintf(loc_phi, " %d:%.9f", j, cur_phi);
		}
		fprintf(loc_phi, "\n");
	}
	fclose(loc_phi);
	memset(file_name, 0, sizeof(char) * 128);
	snprintf(file_name, 128, "data/%s_gl_phi.iter%d", save_prefix.c_str(), iter);
	FILE * gl_phi = fopen(file_name, "w");
	for (int i = 0; i < k_global; ++i) {
		// global_topic的phi
		fprintf(gl_phi, "global_topic_id:%d", i);
		for (int j = 0; j < vocabulary_size; ++j) {
			cur_phi = (double(global_topic_wc_with_wordid[i * vocabulary_size + j]) + beta_global)
					/ (double(global_topic_wc[i]) + vocabulary_size * beta_global);
			fprintf(gl_phi, " %d:%.9f", j, cur_phi);
		}
		fprintf(gl_phi, "\n");
	}
	fclose(gl_phi);
	return 0;
}

int Model::SaveThetaResult(int iter) {
	char file_name[128];
	snprintf(file_name, 128, "data/%s_theta.iter%d", save_prefix.c_str(), iter);
	FILE * theta_file = fopen(file_name, "w");
	double k_alpha_global = k_global * alpha_global;
	double s_gamma = slidding_window_width * gamma;
	double k_alpha_local = k_local * alpha_local;
	for (uint32_t doc_ind = 0; doc_ind < corpus->docs.size(); ++ doc_ind) {
		DocumentPtr cur_doc = corpus->docs[doc_ind];
		fprintf(theta_file,
				"doc_id:%ld sent_num:%lu word_num:%d\n",
				cur_doc->doc_id,
				cur_doc->sentences.size(),
				cur_doc->word_num);
		for (uint32_t sent_ind = 0; sent_ind < cur_doc->sentences.size(); ++ sent_ind) {
			SentencePtr cur_sent = cur_doc->sentences[sent_ind];
			// global theta
			fprintf(theta_file,
					"sent_id:%d length:%lu",
					cur_sent->sent_id,
					cur_sent->words.size());
			for (int gl_id = 0; gl_id < k_global; ++ gl_id) {
				double theta = 0.0;
				// 对于所有的slidding window，计算概率
				for (int win_ind = 0; win_ind < slidding_window_width; ++ win_ind) {
					double cur_p = 1.0;
					cur_p = cur_p * (double(cur_sent->wc_which_window[win_ind]) + gamma) /
							(double(cur_sent->words.size()) + s_gamma);
					cur_p = cur_p
							* (double(cur_doc->gl_wc_in_s_window[sent_ind + win_ind]) + alpha_mix_global)
							/ (double(cur_doc->wc_in_s_window[sent_ind + win_ind]) + alpha_mix_global + alpha_mix_local);
					cur_p = cur_p
							* (double(cur_doc->wc_with_gl_topic[gl_id]) + alpha_global)
							/ (double(cur_doc->wc_global) + k_alpha_global);
					theta = theta + cur_p;
				}
				fprintf(theta_file, " gl_z_%d:%.9f", gl_id, theta);
			}
			// local theta
			for (int loc_id = 0; loc_id < k_local; ++ loc_id) {
				double theta = 0.0;
				// 对于所有的slidding window，计算概率
				for (int win_ind = 0; win_ind < slidding_window_width; ++ win_ind) {
					double cur_p = 1.0;
					cur_p = cur_p
							* (double(cur_sent->wc_which_window[win_ind]) + gamma)
							/ (double(cur_sent->words.size()) + s_gamma);
					cur_p = cur_p
							* (double(cur_doc->loc_wc_in_s_window[sent_ind + win_ind]) + alpha_mix_local)
							/ (double(cur_doc->wc_in_s_window[sent_ind + win_ind]) + alpha_mix_global + alpha_mix_local);
					cur_p = cur_p
							* (double(cur_doc->wc_with_loc_topic[loc_id]) + alpha_local)
							/ (double(cur_doc->wc_local) + k_alpha_local);
					theta = theta + cur_p;
				}
				fprintf(theta_file, " loc_z_%d:%.9f", loc_id, theta);
			}
			fprintf(theta_file, "\n");
		}
	}
	fclose(theta_file);
	return 0;
}

// 不仅保存topic->word的分布，还保存每个word被分配到哪个窗口，global_id和local_id
int Model::SaveTokenResult(int iter) {
	char file_name[128];
	memset(file_name, 0, sizeof(char) * 128);
	snprintf(file_name, 128, "data/%s_token.iter%d", save_prefix.c_str(), iter);
	FILE * result = fopen(file_name, "w");
	for (uint32_t doc_ind = 0; doc_ind < corpus->docs.size(); ++ doc_ind) {
		DocumentPtr cur_doc = corpus->docs[doc_ind];
		fprintf(result,
            "doc_id:%ld sent_num:%lu word_num:%d\n",
				cur_doc->doc_id,
				cur_doc->sentences.size(),
				cur_doc->word_num);
		for (uint32_t sent_id = 0; sent_id < cur_doc->sentences.size(); ++ sent_id) {
			SentencePtr cur_sent = cur_doc->sentences[sent_id];
			fprintf(result,
					"sent_id:%d length:%lu",
					cur_sent->sent_id,
					cur_sent->words.size());
			for (uint32_t word_ind = 0; word_ind < cur_sent->words.size(); ++ word_ind) {
				fprintf(result,
						" %d:%d,%d,%d",
						cur_sent->words[word_ind],
						cur_sent->words_which_window[word_ind],
						cur_sent->words_global_topic[word_ind],
						cur_sent->words_local_topic[word_ind]);
			}
			fprintf(result, "\n");
		}
	}
	fclose(result);
	return 0;
}

// 保存每个topic下的word个数和每个topic下针对该word_id的词的个数
int Model::SaveVocabularyResult(int iter) {
	char file_name[128];
	memset(file_name, 0, sizeof(char) * 128);
	snprintf(file_name, 128, "data/%s_vocabulary.iter%d", save_prefix.c_str(), iter);
	FILE * result = fopen(file_name, "w");
	fprintf(result, "global_topic_num:%d", k_global);
	for (int gl_id = 0; gl_id < k_global; ++ gl_id) {
		fprintf(result, " gl_topic_%d:%d", gl_id, global_topic_wc[gl_id]);
	}
	fprintf(result, "\n");
	fprintf(result, "local_topic_num:%d", k_local);
	for (int loc_id = 0; loc_id < k_local; ++ loc_id) {
		fprintf(result, " loc_topic_%d:%d", loc_id, local_topic_wc[loc_id]);
	}
	fprintf(result, "\n");

	for (int gl_id = 0; gl_id < k_global; ++ gl_id) {
		fprintf(result, "global_topic_id:%d", gl_id);
		for (int w_id = 0; w_id < vocabulary_size; ++ w_id) {
			fprintf(result,
					" %d:%d",
					w_id,
					global_topic_wc_with_wordid[gl_id * vocabulary_size + w_id]);
		}
		fprintf(result, "\n");
	}
	for (int loc_id = 0; loc_id < k_local; ++ loc_id) {
		fprintf(result, "local_topic_id:%d",loc_id);
		for (int w_id = 0; w_id < vocabulary_size; ++ w_id){
			fprintf(result,
					" %d:%d",
					w_id,
					local_topic_wc_with_wordid[loc_id * vocabulary_size] + w_id);
		}
		fprintf(result, "\n");
	}
	fclose(result);
	return 0;
}

int Model::InfGibbsSampling() {
	if (model_gl_topic_wc == NULL
			|| model_gl_topic_wc_with_wordid == NULL
			|| model_loc_topic_wc == NULL
			|| model_loc_topic_wc_with_wordid == NULL) {
		fprintf(stderr,
				"%s %d ERROR: Model data loss!\n",
				__FILE__,
				__LINE__);
		return -1;
	}
	// 一些全局的double；放在循环体外节省时间
	double v_beta_global = vocabulary_size * beta_global;
	double k_alpha_global = k_global * alpha_global;
	double s_gamma = slidding_window_width * gamma;
	double v_beta_local = vocabulary_size * beta_local;
	double k_alpha_local = k_local * alpha_local;

	// doc_count
	for (uint32_t doc_ind = 0; doc_ind < corpus->docs.size() ; ++ doc_ind) {
		// doc_count
		if (doc_ind % 100000 == 0 && doc_ind != 0 && is_debug) {
			fprintf(stderr,
					"%s %d INFO: 100, 000 docs be sampled!\n",
					__FILE__,
					__LINE__);
		}
		DocumentPtr cur_doc = corpus->docs[doc_ind];
		for (uint32_t sent_ind = 0; sent_ind < cur_doc->sentences.size(); ++ sent_ind) {
			SentencePtr cur_sentence = cur_doc->sentences[sent_ind];
			// for each words
			for (uint32_t w_ind = 0; w_ind < cur_sentence->words.size(); ++ w_ind) {
				int cur_word_id = cur_sentence->words[w_ind];
				// which window
				int old_v = cur_sentence->words_which_window[w_ind];
				// which topic
				int old_global_topic = cur_sentence->words_global_topic[w_ind];
				int old_local_topic = cur_sentence->words_local_topic[w_ind];
				// just for exception
				if (old_global_topic >= 0 && old_local_topic >= 0) {
					fprintf(stderr,
							"%s %d ERROR: A word has two topics!\n",
							__FILE__,
							__LINE__);
					return -1;
				}
				// decrease
				if (old_global_topic >= 0) {
					// global_topic
					cur_sentence->words_global_topic[w_ind] = -1;
					cur_sentence->words_which_window[w_ind] = -1;
					cur_sentence->wc_which_window[old_v]--;
					cur_doc->gl_wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window_with_gl_topic[(sent_ind + old_v) * k_global + old_global_topic]--;
					cur_doc->wc_with_gl_topic[old_global_topic]--;
					cur_doc->wc_global--;
					global_topic_wc[old_global_topic]--;
					global_topic_wc_with_wordid[old_global_topic * vocabulary_size + cur_word_id]--;
				} else if (old_local_topic >= 0) {
					// local_topic
					cur_sentence->words_local_topic[w_ind] = -1;
					cur_sentence->words_which_window[w_ind] = -1;
					cur_sentence->wc_which_window[old_v]--;
					cur_doc->loc_wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window[sent_ind + old_v]--;
					cur_doc->wc_in_s_window_with_loc_topic[(sent_ind + old_v) * k_local + old_local_topic]--;
					cur_doc->wc_with_loc_topic[old_local_topic]--;
					cur_doc->wc_local--;
					local_topic_wc[old_local_topic]--;
					local_topic_wc_with_wordid[old_local_topic * vocabulary_size + cur_word_id]--;
				} else {
					fprintf(stderr, "A words has no topics!\n");
					return -1;
				}

				// sampling
				int sample_iter = 0;
				for (int win_id = 0; win_id < slidding_window_width; ++ win_id) {
					for (int gl_id = 0; gl_id < k_global; ++ gl_id) {
						// global_topic
						// l是slidding window id， m是topic id
						double score = 1.0;
						// 此处和训练过程中的GibbsSampling的不同
						int gl_id_with_w_id_ind = gl_id * vocabulary_size + cur_word_id;
						score = score *
								(double(global_topic_wc_with_wordid[gl_id_with_w_id_ind] + model_gl_topic_wc_with_wordid[gl_id_with_w_id_ind]) + beta_global)
								/ (double(global_topic_wc[gl_id]) + double(model_gl_topic_wc[gl_id]) + v_beta_global);
						score = score
								* (double(cur_sentence->wc_which_window[win_id]) + gamma)
								/ (double(cur_sentence->words.size() + s_gamma));
						score = score
								* (double(cur_doc->gl_wc_in_s_window[sent_ind + win_id]) + alpha_mix_global)
								/ (double(cur_doc->wc_in_s_window[sent_ind + win_id]) + alpha_mix_global + alpha_mix_local);
						score = score
								* (double(cur_doc->wc_with_gl_topic[gl_id]) + alpha_global)
								/ (double(cur_doc->wc_global) + k_alpha_global);
						p[sample_iter++] = score;
					}
					for (int loc_id = 0; loc_id < k_local; ++ loc_id) {
						// local_topic
						double score = 1.0;
						int loc_id_with_w_id = loc_id * vocabulary_size + cur_word_id;
						score = score *
								(double(local_topic_wc_with_wordid[loc_id_with_w_id] + model_loc_topic_wc_with_wordid[loc_id_with_w_id]) + beta_local)
								/ (double(local_topic_wc[loc_id] + model_loc_topic_wc[loc_id]) + v_beta_local);
						score = score *
								(double(cur_sentence->wc_which_window[win_id]) + gamma)
								/ (double(cur_sentence->words.size() + s_gamma));
						score = score *
								(double(cur_doc->loc_wc_in_s_window[sent_ind + win_id]) + alpha_mix_local)
								/ (double(cur_doc->wc_in_s_window[sent_ind + win_id]) + alpha_mix_global + alpha_mix_local);
						score = score
								* (double(cur_doc->wc_in_s_window_with_loc_topic[(sent_ind + win_id) * k_local + loc_id]) + alpha_local)
								/ (double(cur_doc->loc_wc_in_s_window[sent_ind + win_id]) + k_alpha_local);
						p[sample_iter++] = score;
					}
				}

				// multinomial采样
				int p_length = (k_global + k_local) * slidding_window_width;
				for (int z = 1; z < p_length; ++z) {
					p[z] = p[z] + p[z - 1];
				}
				double random_p =
						((double) rand() / RAND_MAX) * p[p_length - 1];
				int new_index = 0;
				for (int z = 0; z < p_length; ++z) {
					if (p[z] >= random_p) {
						new_index = z;
						break;
					}
				}

				// 采样完毕, 计算新采样的词频率
				int new_window = new_index / (k_global + k_local);
				int new_topic_index = new_index % (k_global + k_local);
				int new_global_topic = -1;
				int new_local_topic = -1;
				if (new_topic_index < k_global) {
					new_global_topic = new_topic_index;
				} else {
					new_local_topic = new_topic_index - k_global;
				}

				// increase
				cur_sentence->wc_which_window[new_window]++;
				cur_sentence->words_which_window[w_ind] = new_window;
				cur_sentence->words_global_topic[w_ind] = new_global_topic;
				cur_sentence->words_local_topic[w_ind] = new_local_topic;
				if (new_global_topic >= 0) {
					// global_topic
					cur_doc->wc_in_s_window[sent_ind + new_window]++;
					cur_doc->gl_wc_in_s_window[sent_ind + new_window]++;
					cur_doc->wc_in_s_window_with_gl_topic[(sent_ind + new_window) * k_global	+ new_global_topic]++;
					cur_doc->wc_with_gl_topic[new_global_topic]++;
					cur_doc->wc_global++;
					global_topic_wc[new_global_topic]++;
					global_topic_wc_with_wordid[new_global_topic * vocabulary_size + cur_word_id]++;
				} else if (new_local_topic >= 0) {
					// local_topic
					cur_doc->wc_in_s_window[sent_ind + new_window]++;
					cur_doc->loc_wc_in_s_window[sent_ind + new_window]++;
					cur_doc->wc_in_s_window_with_loc_topic[(sent_ind + new_window) * k_local + new_local_topic]++;
					cur_doc->wc_with_loc_topic[new_local_topic]++;
					cur_doc->wc_local++;
					local_topic_wc[new_local_topic]++;
					local_topic_wc_with_wordid[new_local_topic * vocabulary_size + cur_word_id]++;
				} else {
					fprintf(stderr,
							"%s %d ERROR: A word has no new topic!\n",
							__FILE__,
							__LINE__);
					return -1;
				}
			}
		}
	}
	return 0;
}
int Model::Inference() {
	int cur_iter = 0;
	for (; cur_iter < max_iterator; ++cur_iter) {
		if (InfGibbsSampling() != 0) {
			fprintf(stderr,
					"%s %d ERROR: Gibbs Sampling iter %d failed!\n",
					__FILE__,
					__LINE__,
					cur_iter);
			return -1;
		}
		fprintf(stderr,
				"%s %d INFO: Gibbs Sampling iter %d success!\n",
				__FILE__,
				__LINE__,
				cur_iter);
		if ((cur_iter + 1) % (save_iter * 5) == 0) {
			SaveThetaResult(cur_iter);
			SaveTokenResult(cur_iter);
		}
		else if ((cur_iter + 1) % save_iter == 0) {
			SaveThetaResult(cur_iter);
		}
	}
	SaveThetaResult(cur_iter);
	SaveTokenResult(cur_iter);
	return 0;
}
