/*
 * corpus.cc
 *
 */


#include <cstdio>
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include "corpus.h"
#include <iostream>

using namespace boost::property_tree;
const static int MAX_LENG_SIZE = 10240;

int Corpus::Load(const std::string corpus_file_path) {
	FILE * corpus_file = fopen(corpus_file_path.c_str(), "r");
	if (!corpus_file) {
		fprintf(stderr,
            "%s:%d ERROR: Cannot open corpus file:%s!\n",
				__FILE__,
				__LINE__,
                corpus_file_path.c_str());
		return -1;
	}

	int read_doc_num = 0;	//已经读取的doc_num
    char buff[MAX_LENG_SIZE];
    while(fgets(buff, sizeof(buff)-1, corpus_file)){
        try {
            boost::property_tree::ptree pt;
            std::stringstream ss(buff);
            boost::property_tree::read_json(ss, pt);
            
            //解析一篇doc
            DocumentPtr doc = DocumentPtr(new Document());
	        doc->doc_id = pt.get<int64_t>("doc_id", -1);
            if(doc->doc_id <= 0){
                continue;
            }
            boost::property_tree::ptree sentence_pt = pt.get_child("sentences");
            int read_word_num = 0;
            for(boost::property_tree::ptree::iterator it=sentence_pt.begin(); it!=sentence_pt.end(); it++){
                ptree sub_tree = it->second;
                SentencePtr sentence = SentencePtr(new Sentence());
                sentence->sent_id = sub_tree.get("sent_id", -1);
                if(sentence->sent_id <= 0){
                    continue;
                }
                sentence->offset = read_word_num;
                boost::property_tree::ptree word_pt = sub_tree.get_child("words");
                for(boost::property_tree::ptree::iterator it2=word_pt.begin(); it2!=word_pt.end(); it2++){
                    std::string word_id_str = it2->second.data();
                    int word_id = atoi(word_id_str.c_str());
                    sentence->words.push_back(word_id);
                    doc->word_num ++;
                }
                read_word_num += sentence->words.size();
                doc->sentences.push_back(sentence);
            }
            this->docs.push_back(doc);
            read_doc_num ++;
            doc->output();
        }catch(boost::property_tree::ptree_error & e){
            std::cerr << "catch exception while process:" << buff << std::endl;
            continue;
        }

    }//end of while
	return read_doc_num;
}
