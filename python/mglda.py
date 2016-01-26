#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# mg-lda
# This code is available under the MIT License.
# (c)2012 Masanao Ochi.

import time, logging, sys
import numpy
from pyutil.program.log import logging_config

class MGLDA:
    def __init__(self, K_gl, K_loc, gamma, alpha_gl, alpha_loc, alpha_mix_gl, alpha_mix_loc, beta_gl, beta_loc, T, docs, W, smartinit=False):
        
        self.K_gl = K_gl
        self.K_loc = K_loc
        
        self.gamma         = gamma
        self.alpha_gl      = alpha_gl # parameter of topics prior
        self.alpha_loc     = alpha_loc
        self.alpha_mix_gl  = alpha_mix_gl
        self.alpha_mix_loc = alpha_mix_loc
        
        self.beta_gl = beta_gl # parameter of words prior
        self.beta_loc = beta_loc
        
        self.T = T # sliding window width
        
        self.docs = docs
        self.W = W

        self.v_d_s_n     = [] #sumi
        self.r_d_s_n     = [] #sumi
        self.z_d_s_n     = [] #sumi
        
        self.n_gl_z_w    = numpy.zeros((self.K_gl, self.W))
        self.n_gl_z      = numpy.zeros(self.K_gl)
        self.n_d_s_v     = [] #sumi
        self.n_d_s       = [] #sumi
        self.n_d_v_gl    = [] #sumi
        self.n_d_v       = [] #sumi
        self.n_d_gl_z    = numpy.zeros((len(self.docs), self.K_gl))
        self.n_d_gl      = numpy.zeros((len(self.docs)))

        self.n_loc_z_w   = numpy.zeros((self.K_loc, self.W))
        self.n_loc_z     = numpy.zeros(self.K_loc)
        self.n_d_v_loc   = [] #sumi
        self.n_d_v_loc_z = [] #sumi

        self.inflation = 0

        logging.info("random fitting to initialize")
        for m, doc in enumerate(self.docs):
            v_d = []
            r_d = []
            z_d = []
            
            n_d_s_v_d = []
            n_d_s_d = []
            
            n_d_v_gl_v = []
            n_d_v_v = []
            n_d_v_loc_v = []
            n_d_v_loc_z_v = []
            for v in range(self.T+len(doc)-1):
                n_d_v_gl_v.append(self.inflation) # initialize word count with global topic for each sliding window
                n_d_v_v.append(self.inflation) # initialize word count for each sliding window
                n_d_v_loc_v.append(self.inflation) # initialize word count with local topic for each sliding window
                
                n_d_v_loc_z_z = []
                for k in range(self.K_loc):
                    n_d_v_loc_z_z.append(self.inflation) # initialize word count assigned local topic for each sliding window
                n_d_v_loc_z_v.append(n_d_v_loc_z_z) 
            
            self.n_d_v_gl.append(n_d_v_gl_v)
            self.n_d_v.append(n_d_v_v)
            self.n_d_v_loc.append(n_d_v_loc_v)
            
            self.n_d_v_loc_z.append(n_d_v_loc_z_v)
            
            for s, sent in enumerate(doc):
                v_s = []
                r_s = []
                z_s = []
                for i, word in enumerate(sent):
                    v     = numpy.random.randint(0, self.T) # initialize sliding window for each word
                    v_s.append(v)
                    
                    r_int = numpy.random.randint(0, 2) # initialize topic category
                    r = ""
                    if r_int == 0:
                        r = "gl"
                    else:
                        r = "loc"
                    r_s.append(r) 
                    
                    z = 0
                    if r == "gl":
                        z = numpy.random.randint(0, self.K_gl) # initialize global topic
                    else:
                        z = numpy.random.randint(0, self.K_loc) # initialize local topic
                    z_s.append(z)
                v_d.append(v_s)
                r_d.append(r_s)
                z_d.append(z_s)
                
                n_d_s_v_s = []
                for v in range(self.T):
                    n_d_s_v_s.append(self.inflation) # initialize n_d_s_v
                n_d_s_v_d.append(n_d_s_v_s)
                
                n_d_s_d.append(self.inflation) # initialize n_d_s
            
            self.v_d_s_n.append(v_d)
            self.r_d_s_n.append(r_d)
            self.z_d_s_n.append(z_d)
            
            self.n_d_s_v.append(n_d_s_v_d)
            self.n_d_s.append(n_d_s_d)
        
        logging.info("initialize")
        for m, doc in enumerate(self.docs):
            for s, sent in enumerate(doc):
                for i, word in enumerate(sent):
                    v = self.v_d_s_n[m][s][i] # 0--T
                    r = self.r_d_s_n[m][s][i]
                    z = self.z_d_s_n[m][s][i]
                    if r == "gl":
                        self.n_gl_z_w[z][word]      += 1
                        self.n_gl_z[z]              += 1
                        self.n_d_v_gl[m][s+v]       += 1
                        self.n_d_gl_z[m][z]         += 1
                        self.n_d_gl[m]              += 1
                    elif r == "loc":
                        self.n_loc_z_w[z][word]     += 1
                        self.n_loc_z[z]             += 1
                        self.n_d_v_loc[m][s+v]      += 1
                        self.n_d_v_loc_z[m][s+v][z] += 1
                    else:
                        logging.info("error0: %s", str(r))

                    self.n_d_s_v[m][s][v]           += 1
                    self.n_d_s[m][s]                += 1
                    self.n_d_v[m][s+v]              += 1

        logging.info("init comp.")

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            for s, sent in enumerate(doc):
                for i, word in enumerate(sent):
                    v = self.v_d_s_n[m][s][i] # 0--T
                    r = self.r_d_s_n[m][s][i]
                    z = self.z_d_s_n[m][s][i]
                    
                    # discount
                    if r == "gl":
                        self.n_gl_z_w[z][word]      -= 1
                        self.n_gl_z[z]              -= 1
                        self.n_d_v_gl[m][s+v]       -= 1
                        self.n_d_gl_z[m][z]         -= 1
                        self.n_d_gl[m]              -= 1
                    elif r == "loc":
                        self.n_loc_z_w[z][word]     -= 1
                        self.n_loc_z[z]             -= 1
                        self.n_d_v_loc[m][s+v]      -= 1
                        self.n_d_v_loc_z[m][s+v][z] -= 1
                    else:
                        logging.info("error1: %s", str(r))

                    self.n_d_s_v[m][s][v]       -= 1
                    self.n_d_s[m][s]            -= 1
                    self.n_d_v[m][s+v]          -= 1
                    
                    # sampling topic new_z for t
                    p_v_r_z = []
                    label_v_r_z = []
                    for v_t in range(self.T):
                        gl_term2 = float(self.n_d_s_v[m][s][v_t] + self.gamma) / (self.n_d_s[m][s] + self.T*self.gamma)
                        gl_term3 = float(self.n_d_v_gl[m][s+v_t] + self.alpha_mix_gl) / (self.n_d_v[m][s+v_t] + self.alpha_mix_gl + self.alpha_mix_loc)
                        ll_term2 = float(self.n_d_s_v[m][s][v_t] + self.gamma) / (self.n_d_s[m][s] + self.T*self.gamma)
                        ll_term3 = float(self.n_d_v_loc[m][s+v_t] + self.alpha_mix_loc) / (self.n_d_v[m][s+v_t] + self.alpha_mix_gl + self.alpha_mix_loc)

                        # for r == "gl"
                        for z_t in range(self.K_gl):
                            label = [v_t, "gl", z_t]
                            label_v_r_z.append(label)
                            # sampling eq as gl
                            term1 = float(self.n_gl_z_w[z_t][word] + self.beta_gl) / (self.n_gl_z[z_t] + self.W*self.beta_gl)
                            term4 = float(self.n_d_gl_z[m][z_t] + self.alpha_gl) / (self.n_d_gl[m] + self.K_gl*self.alpha_gl)
                            score = term1 * gl_term2 * gl_term3 * term4
                            p_v_r_z.append(score)
                        # for r == "loc" 
                        for z_t in range(self.K_loc):
                            label = [v_t, "loc", z_t]
                            label_v_r_z.append(label)
                            # sampling eq as loc
                            term1 = float(self.n_loc_z_w[z_t][word] + self.beta_loc) / (self.n_loc_z[z_t] + self.W*self.beta_loc)
                            term4 = float(self.n_d_v_loc_z[m][s+v_t][z_t] + self.alpha_loc) / (self.n_d_v_loc[m][s+v_t] + self.K_loc*self.alpha_loc)
                            score = term1 * ll_term2 * ll_term3 * term4
                            p_v_r_z.append(score)
                    
                    np_p_v_r_z = numpy.array(p_v_r_z)
                    new_p_v_r_z_idx = numpy.random.multinomial(1, np_p_v_r_z / np_p_v_r_z.sum()).argmax()
                    new_v, new_r, new_z = label_v_r_z[new_p_v_r_z_idx]
                    
                    # update
                    if new_r == "gl":
                        self.n_gl_z_w[new_z][word]          += 1
                        self.n_gl_z[new_z]                  += 1
                        self.n_d_v_gl[m][s+new_v]           += 1
                        self.n_d_gl_z[m][new_z]             += 1
                        self.n_d_gl[m]                      += 1
                    elif new_r == "loc":
                        self.n_loc_z_w[new_z][word]         += 1
                        self.n_loc_z[new_z]                 += 1
                        self.n_d_v_loc[m][s+new_v]          += 1
                        self.n_d_v_loc_z[m][s+new_v][new_z] += 1
                    else:
                        logging.info("error2: %s", str(r))

                    self.n_d_s_v[m][s][new_v]               += 1
                    self.n_d_s[m][s]                        += 1
                    self.n_d_v[m][s+new_v]                  += 1
                    
                    self.v_d_s_n[m][s][i] = new_v
                    self.r_d_s_n[m][s][i] = new_r
                    self.z_d_s_n[m][s][i] = new_z

    def worddist(self):
        """get topic-word distribution"""
        return (self.n_gl_z_w + 1) / (self.n_gl_z[:, numpy.newaxis] + 1), (self.n_loc_z_w + 1) / (self.n_loc_z[:, numpy.newaxis] + 1)

def mglda_learning(mglda, iteration, voca, out):
    logging.info("start in mglda_learning")
    for i in range(iteration):
        start_str = "==== %s-th inference ====\n" % i
        logging.info(start_str)
        out.write(start_str)
        mglda.inference()
        logging.info("inference %s-th complete, begin to output topic dist", i)
        output_word_topic_dist(mglda, voca, out)
        logging.info("output_word_topic_dist %s-th succeed", i)

def output_word_topic_dist(mglda, voca, out):
    z_gl_count = numpy.zeros(mglda.K_gl, dtype=int)
    z_loc_count = numpy.zeros(mglda.K_loc, dtype=int)
    word_gl_count = [dict() for k in xrange(mglda.K_gl)]
    word_loc_count = [dict() for k in xrange(mglda.K_loc)]
    
    for m, doc in enumerate(mglda.docs):
        for s, sent in enumerate(doc):
            for i, word in enumerate(sent):
#                v = mglda.v_d_s_n[m][s][i] # 0--T
                r = mglda.r_d_s_n[m][s][i]
                z = mglda.z_d_s_n[m][s][i]
                if r == "gl":
                    z_gl_count[z] += 1
                    if word in word_gl_count[z]:
                        word_gl_count[z][word]  += 1
                    else:
                        word_gl_count[z][word]   = 1
                elif r == "loc":
                    z_loc_count[z] += 1
                    if word in word_loc_count[z]:
                        word_loc_count[z][word] += 1
                    else:
                        word_loc_count[z][word]  = 1
                else:
                    logging.error("error3: %s", str(r))

    phi_gl, phi_loc = mglda.worddist()
    for k in range(mglda.K_gl):
        out.write("\n-- global topic: %d (%d words)\n" % (k, z_gl_count[k]))
        out.write("mglda.n_gl_z[k]:%s\n" % mglda.n_gl_z[k])
        for w in numpy.argsort(-phi_gl[k])[:50]:
            out.write("%s: %f (%d)\n" % (voca[w].encode('utf-8'), phi_gl[k,w], word_gl_count[k].get(w,0)))

    for k in range(mglda.K_loc):
        out.write("\n-- local topic: %d (%d words)\n" % (k, z_loc_count[k]))
        out.write("mglda.n_loc_z[k]:%s\n" % mglda.n_loc_z[k])
        for w in numpy.argsort(-phi_loc[k])[:50]:
            out.write("%s: %f (%d)\n" % (voca[w].encode('utf-8'), phi_loc[k,w], word_loc_count[k].get(w,0)))

def test():
    data_file = sys.argv[1]
    model_file = sys.argv[2]
#    import nltk.corpus
    logging.info("begin to run")
    t1 = time.time()
    import vocabulary_for_mglda as vocabulary
    
    #corpus = vocabulary.load_corpus_each_sentence("pushed_words.dat")
    corpus = vocabulary.load_corpus_each_sentence(data_file)
    t2 = time.time()
    logging.info("load corpus succeed. cost:%d s", t2-t1)

    #docs[sentence_idx][word_idx]
    voca = vocabulary.Vocabulary(True)
    docs = [voca.doc_to_ids_each_sentence(doc) for doc in corpus]
    t3 = time.time()
    logging.info("doc_to_id succeed. cost:%d s", t3-t2)

    K_gl, K_loc, gamma, alpha_gl, alpha_loc, alpha_mix_gl, alpha_mix_loc, beta_gl, beta_loc, T, docs, W = 50, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3, docs, voca.size()
    mglda = MGLDA(K_gl, K_loc, gamma, alpha_gl, alpha_loc, alpha_mix_gl, alpha_mix_loc, beta_gl, beta_loc, T, docs, W)
    logging.info("corpus=%d, words=%d, K_gl=%d, K_loc=%d, gamma=%f, alpha_gl=%f, alpha_loc=%f, alpha_mix_gl=%f, alpha_mix_loc=%f, beta_gl=%f, beta_loc=%f" % (len(corpus), len(voca.vocas), K_gl, K_loc, gamma, alpha_gl, alpha_loc, alpha_mix_gl, alpha_mix_loc, beta_gl, beta_loc))
    t4 = time.time()
    logging.info("initialize succeed. cost:%d s", t4-t3)
    logging.info("begin to learn")
    
    out = open(model_file, 'wb')
    iteration = 1000
    mglda_learning(mglda, iteration, voca, out)
    out.close()
    logging.info("learn succeed. cost:%d s", time.time()-t4)

if __name__ == "__main__":
    logging_config(log_file="xxx.log", log_level=logging.INFO)
    test()
