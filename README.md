## For what
Understanding the Application about Deep Learning in Text Matching Area & Implement Codes about the Classical Methods


## People in these area
- [Po-Sen Huang](https://posenhuang.github.io/full_publication.html)
- [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)
- [Richard Socher](http://www.socher.org/index.php/Main/HomePage)
- [Hang Li](http://www.hangli-hl.com/index.html)



## Methods & Papers

> [**DSSM**](./DSSM/dssm.py)
<br> [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)
<br> CIKM 2013
<br> 词袋模型,基于语义表达的结构, word hash + DNN 

> [**CDSSM**]() 
 <br> [Learning Semantic Representations Using Convolutional Neural Networks for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)
 <br> WWW 2014
 
> [**CLSM**]() 
 <br> [A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2014_cdssm_final.pdf)
 <br> CIKM 2014
 <br> 基于匹配的结构, word hash + CNN, CLSM和C-DSSM有什么区别呢
 
> [**DSSM的应用**]()  
[Modeling Interestingness with Deep Neural Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/10/604_Paper.pdf)
<br> EMNLP 2014
<br> DSSM应用于文本分析，在automatic highlighting和contextual entity search问题上效果好。
<br> 主要有两点贡献：
<br> 1) DSSM + CNN
<br> 2) 不针对相关性，加了一个ranker

> [**ARC-I/ARC-II**]()   
  [Convolutional Neural Network Architectures 
for Matching Natural Language Sentences](https://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)
<br> NIPS 2014
<br> CNN的基于语义表达和基于匹配的两种结构; 增加了门解决句子长度不一致问题

> [**CNTN**]() 
<br> [Convolutional Neural Tensor Network 
Architecture for Community-based Question Answering](https://ijcai.org/Proceedings/15/Papers/188.pdf)
<br> IJCAI 2015
<br> (D)CNN+MLP(tensor layer); 基于语义表达的结构


## Related talks and papers
[Deep Learning for Web Search and
Natural Language Processing](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/wsdm2015.v3.pdf)



## Downloads 
[DSSM/Sent2Vec Release Version](https://www.microsoft.com/en-us/download/details.aspx?id=52365)
<br> MSRA发布的Sent2Vec发行版