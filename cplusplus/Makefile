.PHONY: clean

BIN_TARGET=../bin/mglda
CXX=g++
CFLAGS = -g -O2 -fno-omit-frame-pointer -Wall -Wno-reorder -DHAVE_NETINET_IN_H -DHAVE_INTTYPES_H -pthread -DLINUX -std=c++11
CXXFLAGS = $(CFLAGS) -Wunknown-pragmas

BASE_LIBS = -lz -lcrypto -lpthread -ldl -lboost_system -lboost_thread


LIBS = -Xlinker "-("  \
    $(BASE_LIBS)   \
    -Xlinker "-)"

LDFLAGS = $(LIBS)

SRCS=$(wildcard *.cc)
OBJS = $(SRCS:.cc=.o)
DEPS = $(SRCS:.cc=.d)

all: $(BIN_TARGET)

$(BIN_TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

%.o:%.cc Makefile
	$(CXX) -c  $(CXXFLAGS) $*.cc -o $@
	$(CXX) -MM $(CXXFLAGS) $*.cc > $*.d

clean:
	rm -rf *.o *.d $(BIN_TARGET) 
