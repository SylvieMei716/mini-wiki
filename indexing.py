'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict
import json
import os
import gzip


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter()  # token count
        self.vocabulary = set()  # the vocabulary of the collection
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}

        self.index = defaultdict(list)  # the index

    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics = {'unique_token_count': 0,
                          'total_token_count': 0,
                          'number_of_documents': 0,
                          'mean_document_length': 0, 
                          'index_type':'BasicInvertedIndex', 
                          'vocab': Counter()}
        
    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # Check if the document exists in the metadata
        if docid not in self.document_metadata:
            return
        
        for term in list(self.index.keys()):
            self.index[term] = [(d, count) for (d, count) in self.index[term] if d != docid]
            if not self.index[term]:
                del self.index[term]
                
        del self.document_metadata[docid]
                
        # Update statistics
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] = sum(doc_data['length'] for doc_data in self.document_metadata.values())
        self.statistics['number_of_documents'] = len(self.document_metadata)
        self.statistics['mean_document_length'] = (
            self.statistics['total_token_count'] / self.statistics['number_of_documents']
        ) if self.document_metadata else 0

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        term_counts = Counter([term for term in tokens if term is not None])
        self.document_metadata[docid] = {
            "length": len(tokens),
            "unique_tokens": len(term_counts)
        }
        
        for term, count in term_counts.items():
            if term is not None:
                self.vocabulary.add(term)
                self.index[term].append((docid, count))
                self.statistics['vocab'][term] += count
                
        # Update statistics after adding the document
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] = sum(doc_data['length'] for doc_data in self.document_metadata.values())
        self.statistics['number_of_documents'] = len(self.document_metadata)
        self.statistics['mean_document_length'] = (
            self.statistics['total_token_count'] / self.statistics['number_of_documents']
        ) if self.document_metadata else 0

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        postings = self.get_postings(term)
        term_metadata = {
            "num_docs": len(postings),
            "total_term_freq": sum([freq[1] for freq in postings])
        }
        
        return term_metadata
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        return {key: value for key, value in self.statistics.items() if key not in ['index_type', 'vocab']}

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        os.makedirs(index_directory_name, exist_ok=True)
        with open(f"{index_directory_name}/index.json", 'w') as f:
            json.dump({
                "index": self.index,
                "vocabulary": list(self.vocabulary),
                "document_metadata": self.document_metadata,
                "statistics": self.statistics
            }, f)
            # json.dump(self.index, f)

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        os.makedirs(index_directory_name, exist_ok=True)
        with open(f"{index_directory_name}/index.json", 'r') as f:
            data = json.load(f)
            self.index = defaultdict(list, data["index"])
            self.vocabulary = set(data["vocabulary"])
            self.document_metadata = data["document_metadata"]
            self.statistics = data["statistics"]
            # self.index = json.load(f)


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics = {'unique_token_count': 0,
                          'total_token_count': 0,
                          'number_of_documents': 0,
                          'mean_document_length': 0, 
                          'index_type':'PositionalInvertedIndex', 
                          'vocab': Counter()}
        
    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # Check if the document exists in the metadata
        if docid not in self.document_metadata:
            return

        # Iterate through the index to remove the document from the posting lists
        terms_to_remove = []
        for term, postings in self.index.items():
            # Filter out postings for the document being removed
            new_postings = [(doc, count, pos_lst) for doc, count, pos_lst in postings if doc != docid]
            if new_postings:
                # Update the posting list for this term
                self.index[term] = new_postings
            else:
                # If the term no longer has any postings, mark it for removal
                terms_to_remove.append(term)
        
        # Remove terms that no longer have any postings
        for term in terms_to_remove:
            del self.index[term]
            self.vocabulary.remove(term)  # Remove the term from the vocabulary
        
        del self.document_metadata[docid]
        
        # Update statistics
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] = sum(doc_data['length'] for doc_data in self.document_metadata.values())
        self.statistics['number_of_documents'] = len(self.document_metadata)
        self.statistics['mean_document_length'] = (
            self.statistics['total_token_count'] / self.statistics['number_of_documents']
        ) if self.document_metadata else 0

        # Finally, remove the document metadata
        # del self.document_metadata[docid]

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        term_counts = Counter([term for term in tokens if term is not None])
        self.document_metadata[docid] = {
            "length": len(tokens),
            "unique_tokens": len(term_counts)
        }
        
        for term, count in term_counts.items():
            if term is not None:
                position_lst = [idx for idx, token in enumerate(tokens) if token == term]
                
                posting = (docid, count, position_lst)
                
                term_freq_lst = self.index.setdefault(term, [])
                term_freq_lst.append(posting)
                
                self.vocabulary.add(term)
                
        # Update statistics after adding the document
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] = sum(doc_data['length'] for doc_data in self.document_metadata.values())
        self.statistics['number_of_documents'] = len(self.document_metadata)
        self.statistics['mean_document_length'] = (
            self.statistics['total_token_count'] / self.statistics['number_of_documents']
        ) if self.document_metadata else 0

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        postings = self.get_postings(term)
        term_metadata = {
            "num_docs": len(postings),
            "total_term_freq": sum([freq[1] for freq in postings])
        }
        
        return term_metadata
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        return {key: value for key, value in self.statistics.items() if key not in ['index_type', 'vocab']}

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        os.makedirs(index_directory_name, exist_ok=True)
        with open(f"{index_directory_name}/index.json", 'w') as f:
            json.dump({
                "index": self.index,
                "vocabulary": list(self.vocabulary),
                "document_metadata": self.document_metadata,
                "statistics": self.statistics
            }, f)
            # json.dump(self.index, f)

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        os.makedirs(index_directory_name, exist_ok=True)
        with open(f"{index_directory_name}/index.json", 'r') as f:
            data = json.load(f)
            self.index = defaultdict(list, data["index"])
            self.vocabulary = set(data["vocabulary"])
            self.document_metadata = data["document_metadata"]
            self.statistics = data["statistics"]
            # self.index = json.load(f)

# import tqdm
class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index

        '''
         # TODO (HW3): This function now has an optional argument doc_augment_dict; check README.md
       
        # HINT: Think of what to do when doc_augment_dict exists, how can you deal with the extra information?
        #       How can you use that information with the tokens?
        #       If doc_augment_dict doesn't exist, it's the same as before, tokenizing just the document text
          
        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        #       For HW3, only the BasicInvertedIndex is required to be supported

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input
                      
        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency
        
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        global_token_count = Counter()
        docs = []

        open_file = gzip.open if dataset_path.endswith('.gz') else open
        
        with open_file(dataset_path, 'rt', encoding='utf-8') as file:
            
            for doc_count, line in enumerate(file):
                if 0 <= max_docs == doc_count:
                    break

                document = json.loads(line)
                doc_id = document.get("docid")
                text = document.get(text_key, "")
                tokens = document_preprocessor.tokenize(text)
                
                if doc_augment_dict and doc_id in doc_augment_dict:
                    augment_queries = doc_augment_dict[doc_id]
                    for query in augment_queries:
                        tokens += document_preprocessor.tokenize(query)

                if stopwords:
                    tokens = [None if token in stopwords else token for token in tokens]
                
                global_token_count.update(tokens)

                docs.append((doc_id, tokens))
        
        for doc_id, tokens in docs:
            filtered_tokens = [token if global_token_count[token] >= minimum_word_frequency else None for token in tokens]

            index.add_doc(doc_id, filtered_tokens)
            
        return index


'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')
