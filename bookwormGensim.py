import gensim
import bookwormDB
import subprocess
from collections import defaultdict
from six import iteritems, itervalues, string_types
import logging
logging.basicConfig(level=20)
from gensim.models.word2vec import Word2Vec
import ConfigParser



"""
Some functions to use the bookworm file configuration to easily export.
"""

def fetch_vocab(limit=100000,db=None):
    """
    Fetches the information gensim needs about the vocabulary.
    """
    if db is None:
        db = bookwormDB.CreateDatabase.DB().dbname
    # This would be better implemented as a pair of API calls.
    counts = db.query("SELECT casesens,count FROM words LIMIT %d" % limit ).fetchall()
    books = db.query("SELECT COUNT(*) FROM fastcat WHERE nwords>0").fetchall()[0][0]
    return {"counts":counts,"total_books":books}
            

"""
Patch the gensim class so it can use a pre-created vocabulary,
since we've already done that for bookworm anyway.
"""

class Bookworm2Vec(Word2Vec):
    """
    A patched version of a Word2Vec gensim class that uses a Bookworm database
    as a backend.
    """
    
    def load_vocab(self,keep_raw_vocab=False):
        """
        To load the vocab, we now just run some SQL queries on the 
        Bookworm backend.
        """
        self.import_vocab()
        self.scale_vocab(keep_raw_vocab)
        self.finalize_vocab()

    def import_vocab(self,limit=200000):
        """
        It may be impractical to run on 1,000,000 tokens. I arbitrarily limit to 
        the top 200,000 (case sensitive). 
        Case insensitive would be nice. (?)
        """
        vocab = defaultdict(int)
        data = fetch_vocab(limit=limit)
        for line in data["counts"]:
            (word,count) = line
            if count is not None:
                vocab[word] = count
        sentence_no = data['total_books']
        self.corpus_count = sentence_no
        self.raw_vocab = vocab

class SentenceGenerator():
    """
    Using the command-line bookworm format to ensure it's tokenized.
    It would be better if one of the classes in `tokenize.py` just returned an iterable.
    """
    def __init__(self,limit):
        if limit == float("inf"):
            self.filesim = subprocess.Popen(["bookworm tokenize text_stream | bookworm tokenize token_stream"],shell=True,stdout=subprocess.PIPE,cwd="../../..")
        else:
            self.filesim = subprocess.Popen(["bookworm tokenize text_stream | bookworm tokenize token_stream | head -%d" %limit],shell=True,stdout=subprocess.PIPE,cwd="../../..")
    def __iter__(self):
        """
        The 'token_stream' method means that we get high-quality tokens just by splitting on space.
        """
        for line in self.filesim.stdout:
            yield line.split(" ")

def train_word2vec(limit=float("inf"),workers=6):
    """
    These defaults--6 workers, 10=word window, skip-gram model--are chosen kind of at random. They seem reasonable, though.
    """
    model = PatchedVec(workers=workers,sample=1e-4,size=500,window=10)
    model.load_vocab()
    model.corpus_count = min([limit,model.corpus_count])
    model.train(SentenceGenerator(limit=limit))
    return model


if __name__=="__main__":
    model = main(limit=10000,workers=7)
    
