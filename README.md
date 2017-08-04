## For what
Understanding the Methods in Text Matching Area Including Key-words based Matching Model & Latent Semantic Matching Model.
Implement the Classical Methods.

## Categories
- tradition model (feature based models)
    - Key-words based methods
        - tf-idf model
        - words common rate model
        - find the most important word with adding syntax information
    - boosting models
    - linear models
    - factorization machine
- Semantic deep model
    - representation-based models
        - DSSM, CDSSM
    - interaction-based models


## People in these area
- [Po-Sen Huang](https://posenhuang.github.io/full_publication.html)
- [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)
- [Richard Socher](http://www.socher.org/index.php/Main/HomePage)
- [Hang Li](http://www.hangli-hl.com/index.html)

## Survey
> [深度文本匹配综述(A Survey on Deep Text Matching)](http://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CAPJLAST&filename=JSJX20160920002&uid=WEEvREcwSlJHSldRa1FhdXNXYXJvK0FZMlhXUDZsYnBMQjhHTElMeE1jRT0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4ggI8Fm4gTkoUKaID8j8gFw!!&v=MzA2OTFscVdNMENMTDdSN3FlWU9ac0ZDcmxWYnZPSTFzPUx6N0Jkckc0SDlmTXBvMUZaT3NOWXc5TXptUm42ajU3VDNm)
<br> 


## Methods & Papers about Semantic Methods

> [**DSSM**](./DSSM/dssm.py)
<br> [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)
<br> CIKM 2013
<br> 词袋模型,基于语义表达的结构, word hash + DNN 
<br> [详细解释](http://www.leiphone.com/news/201607/TRldqYnzm6nRbEnY.html)
<br> [代码](./DSSM/dssm.py)
-----
> [**CDSSM**]() 
 <br> [Learning Semantic Representations Using Convolutional Neural Networks for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)
 <br> WWW 2014, word hash + CNN + DNN
----

> [**CLSM**]() 
 <br> [A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2014_cdssm_final.pdf)
 <br> CIKM 2014
 <br> 基于匹配的结构, word hash + CNN, CLSM和C-DSSM有什么区别呢
----
 
> [**DSSM的应用**]()  
[Modeling Interestingness with Deep Neural Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/10/604_Paper.pdf)
<br> EMNLP 2014
<br> DSSM应用于文本分析，在automatic highlighting和contextual entity search问题上效果好。
<br> 主要有两点贡献：
<br> 1) DSSM + CNN
<br> 2) 不针对相关性，加了一个ranker
----

> [**ARC-I/ARC-II**]()   
  [Convolutional Neural Network Architectures 
for Matching Natural Language Sentences](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)
<br> NIPS 2014
<br> CNN的基于语义表达和基于匹配的两种结构; 增加了门解决句子长度不一致问题
----
> [**CNTN**]() 
<br> [Convolutional Neural Tensor Network 
Architecture for Community-based Question Answering](https://ijcai.org/Proceedings/15/Papers/188.pdf)
<br> IJCAI 2015
<br> (D)CNN+MLP(tensor layer); 
<br> 基于语义表达的结构

-----
> [**DeepMatch**]()
<br> [A Deep Architecture for Matching Short Texts](https://papers.nips.cc/paper/5019-a-deep-architecture-for-matching-short-texts.pdf)
<br> NIPS 2013
<br> [Reviews](https://media.nips.cc/nipsbooks/nipspapers/paper_files/nips26/reviews/697.html)
<br> 目的：建模更复杂的匹配关系。最早的基于匹配的结构把。
<br> 结合了localness和hierarchy intrinsic，基于点积的网络不好做的，最大的亮点是用话题模型建立网络吧。

------
> [**DeepMatch_tree**]()
<br> [Syntax-based Deep Matching of Short Texts](https://arxiv.org/pdf/1503.02427.pdf)

## Methods & Papers about Key Words Based Methods
> [****]()
<br> []()

## Related talks and books
* [Deep Learning for Web Search and
Natural Language Processing](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wsdm2015.v3.pdf)
* [Deep Learning for Information Retrieval(Sigir 2016 Tutorial)](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/deep_learning_for_information_retrieval.pdf)
* [Semantic Matching in Search (Sigir 2014 Workshop)](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/semantic_matching_in_search.pdf)
* [Semantic Matching in Search (Book 2014)](http://www.bigdatalab.ac.cn/~junxu/publications/SemanticMatchingInSearch_2014.pdf)
* [gensim notebook](https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks)


## Downloads 
> [DSSM/Sent2Vec Release Version](https://www.microsoft.com/en-us/download/details.aspx?id=52365)
<br> MSRA发布的Sent2Vec发行版

## Datasets
* [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](http://arxiv.org/abs/1502.05698 "Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush") ([fb.ai/babi](http://fb.ai/babi))
* [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340 "Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, Phil Blunsom") ([github.com/deepmind/rc-data](https://github.com/deepmind/rc-data))
* [One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](http://arxiv.org/abs/1312.3005 "Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, Tony Robinson") ([github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark))
* [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909 "Ryan Lowe, Nissan Pow, Iulian Serban, Joelle Pineau") ([cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0](http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/))
* [Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](http://arxiv.org/abs/1506.06724 "Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler") ([BookCorpus](http://www.cs.toronto.edu/~mbweb/))
* [Every publicly available Reddit comment, for research.](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/ "Stuck_In_the_Matrix")
* [Stack Exchange Data Dump](https://archive.org/details/stackexchange "Stack Exchange")
* [Europarl: A Parallel Corpus for Statistical Machine Translation](http://www.iccs.inf.ed.ac.uk/~pkoehn/publications/europarl-mtsummit05.pdf "Philipp Koehn") ([www.statmt.org/europarl/](http://www.statmt.org/europarl/))
* [RTE Knowledge Resources](http://aclweb.org/aclwiki/index.php?title=RTE_Knowledge_Resources)
* [**Kaggle Quora Question Pairs**]()


## Competition
* [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
<br> [SQuAD(The Stanford Question Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/)

## Pretrained Models
* [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo "Berkeley Vision and Learning Center")
* [word2vec](https://code.google.com/p/word2vec/ "Tomas Mikolov")
  * [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
  * [freebase-vectors-skipgram1000.bin.gz](https://docs.google.com/file/d/0B7XkCwpI5KDYaDBDQm1tZGNDRHc/edit?usp=sharing)
* [GloVe](http://nlp.stanford.edu/projects/glove/ "Jeffrey Pennington, Richard Socher, Christopher D. Manning")
* [SENNA](http://ronan.collobert.com/senna/ "R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, P. Kuksa")

## Important Online Courses
* [Stanford CS224d Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
* [Stanford CS20SI Tensorflow for Deep Learning Research](https://web.stanford.edu/class/cs20si/)
* [Stanford CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)


## References
https://github.com/robertsdionne/neural-network-papers/blob/master/README.md
