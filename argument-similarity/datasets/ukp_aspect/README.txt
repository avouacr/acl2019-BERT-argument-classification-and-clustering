UKP ASPECT Corpus
-------------------------------------

The UKP ASPECT Corpus includes 3,595 sentence pairs 
over 28 controversial topics. The sentences were 
crawled from a large web crawl and identified as 
arguments for a given topic using the ArgumenText 
system. The sampling and matching of the sentence 
pairs is described in the paper.
Then, the argument similarity annotation was done via 
crowdsourcing. Each crowd worker could choose from
four annotation options (the exact guidelines are
provided in the Appendix of the paper):

- Different Topic/Can’t decide (DTORCD): Either one or 
both of the sentences belong to a topic different than 
the given one, or you can’t understand one or both 
sentences. If you choose this option, you need to very 
briefly explain, why you chose it (e.g.“The second 
sentence is not grammatical”, “The first sentence is
from a different topic” etc.). 

- No Similarity (NS): The two arguments belong to the 
same topic, but they don’t show any similarity, i.e. 
they speak aboutcompletely different aspects of the topic

- Some Similarity (SS): The two arguments belong to the 
same topic, showing semantic similarity on a few aspects, 
but thecentral message is rather different, or one 
argument is way less specific than the other

- High Similarity (HS): The two arguments belong to the 
same topic, and they speak about the same aspect, e.g. 
using different words

Each line includes the following information:

* topic: the topic keywords used to retrieve the documents

* sentence_1: field for the first sentence

* sentence_2: field for the second sentence

* label: the consolidated crowdsourced gold-standard 
annotation of the sentence pair (DTORCD, NS, SS, HS)


Citation
--------

If you find the data useful, please cite the following paper: 

Nils Reimers, Benjamin Schiller, Tilman Beck, Johannes Daxenberger, Christian Stab, Iryna Gurevych. Classification and Clustering of Arguments with Contextualized Word Embeddings. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), August 2019.

@inproceedings{reimers2019-contextargument,
    author       = {Nils Reimers and Benjamin Schiller and Tilman Beck and Johannes Daxenberger and Christian Stab and Iryna Gurevych},
    title        = {Classification and Clustering of Arguments with Contextualized Word Embeddings},
    booktitle    = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)},
    year         = {2019},
}


Licence
-------

The annotations are released under the 
Creative Commons Attribution-NonCommercial licence (CC BY-NC).