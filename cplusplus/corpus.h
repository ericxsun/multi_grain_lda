/*
 * corpus.h
 *
 */

#ifndef CORPUS_H_
#define CORPUS_H_

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <boost/shared_ptr.hpp>

class Sentence {
public:
	int sent_id;	//句子的id
    std::vector<uint32_t> words;
	int offset; // 句子在doc中的offset，以token为单位
    std::vector<int> words_which_window;	// words属于哪个window; v_d_s_n
    std::vector<int> words_local_topic; // words是哪个local_topic; 如果-1表示其是global_topic, r_d_s_n
    std::vector<int> words_global_topic; // words是那个gl_topic；如果-1表示其是local_topic, z_d_s_n
    std::vector<int> wc_which_window; // 在某个句子中，words属于某个window的有几个; n_d_s_v

public:
	Sentence() {
		sent_id = 0;
		offset = 0;
	}
    void ResetData(int slidding_window_width){
        for(uint32_t i=0; i<words.size(); i++){
            this->words_which_window.push_back(0);
            this->words_local_topic.push_back(0);
            this->words_global_topic.push_back(0);
        }
        for(int i=0; i<slidding_window_width; i++){
            this->wc_which_window.push_back(0);
        }
    }
	~Sentence() {}	
};
typedef boost::shared_ptr<Sentence> SentencePtr;

class Document {
public:
	int64_t doc_id;
    int word_num;
	int sliding_window_num;

public:
    std::vector<SentencePtr> sentences;
	// 分配到某个滑动窗口中的word_count, n_d_v
    std::vector<int> wc_in_s_window;
	// 分配到某个滑动窗口中，topic类型为gl的word_count, n_d_v_gl
    std::vector<int> gl_wc_in_s_window;
	// 分配到某个滑动窗口中，topic类型为loc的word_count, n_d_v_loc
    std::vector<int> loc_wc_in_s_window;
	// 分配到某个滑动窗口中，gl_topic_id的word_count
    std::vector<int> wc_in_s_window_with_gl_topic;
	// 分配到某个滑动窗口中，loc_topic_id的word_count, n_d_v_loc_z
    std::vector<int> wc_in_s_window_with_loc_topic;
	// 分配到某个global_topic中的word_count, n_d_gl_z
    std::vector<int> wc_with_gl_topic;
	// 分配到某个local_topic中的word_count
    std::vector<int> wc_with_loc_topic;

	int wc_global; // n_d_gl
	int wc_local;

public:
    Document() {
        doc_id = 0;
        word_num = 0;
		sliding_window_num = 0;
		wc_global = 0;
		wc_local = 0;
	}

    void ResetData(int k_global, int k_local){
        for(int i=0; i<sliding_window_num; i++){
            gl_wc_in_s_window.push_back(0);
            loc_wc_in_s_window.push_back(0);
            wc_in_s_window.push_back(0);
        }
        for(int i=0; i<sliding_window_num*k_global; i++){
            wc_in_s_window_with_gl_topic.push_back(0);
        }
        for(int i=0; i<sliding_window_num*k_local; i++){
            wc_in_s_window_with_loc_topic.push_back(0);
        }
        for(int i=0; i<k_global; i++){
            wc_with_gl_topic.push_back(0);
        }
        for(int i=0; i<k_local; i++){
            wc_with_loc_topic.push_back(0);     
        }
    }

    void output(){
        std::ostringstream oss;
        oss << "doc_id" << doc_id << ", sentence num:" << sentences.size();
        for(uint32_t i=0; i<sentences.size(); i++){
            oss << ", sentence[";
            SentencePtr sentence = sentences[i];
            for(uint32_t j=0; j<sentence->words.size(); j++){
                oss << sentence->words[j] << " ";
            }
            oss << "]";
        }
        std::cout << "doc info:" << oss.str() << std::endl;
    }

	~Document() {}

};
typedef boost::shared_ptr<Document> DocumentPtr;

class Corpus {
public:
    std::vector<DocumentPtr> docs;
public:
	Corpus() {}
	~Corpus() {}
	int Load(const std::string corpus_file_path);
};
typedef boost::shared_ptr<Corpus> CorpusPtr;

#endif /* CORPUS_H_ */
