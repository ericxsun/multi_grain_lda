/*
 * strtools.cc
 *
 */

#include <cstring>
#include <string>
#include <cstdio>

#include "strtools.h"

int StrSplit(const std::string & input,
		const char split_border,
		std::vector<std::string> * result) {
	result->clear();

    if (input.empty()) {
        return 0;
    }
    std::string tmp;
    std::string::size_type pos_begin = 0;
    std::string::size_type comma_pos = 0;

    while (pos_begin != std::string::npos) {
        comma_pos = input.find(split_border, pos_begin);
        if (comma_pos != std::string::npos) {
        	tmp = input.substr(pos_begin, comma_pos - pos_begin);
            pos_begin = comma_pos + 1;
        } else {
            tmp = input.substr(pos_begin);
            pos_begin = comma_pos;
        }
        result->push_back(tmp);
    }
    return result->size();
}
