"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        # TODO (HW4): If the user has indicated we should use feedback,
        #  create the pseudo-document from the specified number of pseudo-relevant results.
        #  This document is the cumulative count of how many times all non-filtered words show up
        #  in the pseudo-relevant documents. See the equation in the write-up. Be sure to apply the same
        #  token filtering and normalization here to the pseudo-relevant documents.

        # TODO (HW4): Combine the document word count for the pseudo-feedback with the query to create a new query
        # NOTE (HW4): Since you're using alpha and beta to weight the query and pseudofeedback doc, the counts
        #  will likely be *fractional* counts (not integers) which is ok and totally expected.

        """
        if query == "":
            return []
        # 1. Tokenize query
        query_tokens = self.tokenize(query)
        if self.stopwords:
            query_tokens = [token if token not in self.stopwords else None for token in query_tokens]
        query_word_counts = Counter(query_tokens)

        # 2. Fetch a list of possible documents from the index
        doc_word_counts = defaultdict(Counter)
        possible_docs = set()
        for token in query_tokens:
            postings = self.index.get_postings(token)
            if postings:
                for posting in postings:
                    doc_id = posting[0]
                    term_freq = posting[1]
                    doc_word_counts[doc_id][token] = term_freq
                    possible_docs.add(doc_id)

        # 3. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        initial_scores = []
        for doc_id in possible_docs:
            score = self.scorer.score(doc_id, doc_word_counts[doc_id], query_word_counts)
            initial_scores.append((doc_id, score))
            
        initial_scores = sorted(initial_scores, key=lambda x:x[1],reverse=True)
        
        if pseudofeedback_num_docs > 0:
            # print("processing pseudofeedback")
            top_docs = initial_scores[:pseudofeedback_num_docs]
        
            # Adding word vector from relevant docs texts
            extra_word_counts = defaultdict(Counter)
            for doc_id, _ in top_docs:
                if self.raw_text_dict:
                    if doc_id in self.raw_text_dict:
                        raw_text = self.raw_text_dict[doc_id]
                        tokens = self.tokenize(raw_text)
                        if self.stopwords:
                            tokens = [token if token not in self.stopwords else None for token in tokens]
                        extra_word_counts[doc_id] = Counter(tokens)
                        
            adjusted_query_vector = defaultdict(float)
            # Adding alpha * original query vector
            for term, weight in query_word_counts.items():
                adjusted_query_vector[term] += pseudofeedback_alpha * weight
            
            # Adding beta * doc vector / pseudofeedback_num_docs for each relevant doc
            for doc_id, _ in top_docs:
                for term, weight in extra_word_counts[doc_id].items():
                    adjusted_query_vector[term] += pseudofeedback_beta * weight / pseudofeedback_num_docs
            
            # Reranking based on adjusted_query_vector
            doc_word_counts = defaultdict(Counter)
            possible_docs = set()
            for token in adjusted_query_vector:
                postings = self.index.get_postings(token)
                if postings:
                    for posting in postings:
                        doc_id = posting[0]
                        term_freq = posting[1]
                        doc_word_counts[doc_id][token] = term_freq
                        possible_docs.add(doc_id)
                    
            doc_scores = []
            for doc_id in possible_docs:
                score = self.scorer.score(doc_id, doc_word_counts[doc_id], adjusted_query_vector)
                doc_scores.append((doc_id, score))
            doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
            # print("returning updated scores", doc_scores[:5])
                
        # 4. Return **sorted** results as format [(100, 0.5), (10, 0.2), ...]
            return doc_scores
        # print("returning initial scores", initial_scores[:5])
        
        return initial_scores


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # NOTE: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        score = 0
        
        for word, query_count in query_word_counts.items():
            if word in doc_word_counts:
                doc_count = doc_word_counts[word]
                score += query_count * doc_count
                
        # 2. Return the score
        return score


# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid).get('length', 0)
        # doc_len = self.index.get_doc_metadata(docid)['length']
        mu = self.parameters['mu']

        # 2. Compute additional terms to use in algorithm
        score = 0
        for q_term in query_word_counts:
            if q_term and q_term in self.index.index:
                postings = self.index.get_postings(q_term)
                doc_tf = doc_word_counts[q_term] # document TF
                
                if doc_tf > 0:
                    query_tf = query_word_counts[q_term]
                    p_wc = sum([doc[1] for doc in postings]) / \
                        self.index.get_statistics()['total_token_count']
                    tfidf = np.log(1 + doc_tf / (mu * p_wc))
                    
                    score += (query_tf * tfidf)
                    
        # 3. For all query_parts, compute score            
        score = score + len(query_word_counts) * np.log(mu / (doc_len + mu))

        # 4. Return the score
        return score


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid).get('length', 0)
        # doc_len = self.index.get_doc_metadata(docid)['length']
        avg_doc_len = self.index.get_statistics()['mean_document_length']
        N = self.index.get_statistics()['number_of_documents']
        score = 0
        
        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score   
        for word in query_word_counts:
            if word not in doc_word_counts.keys():
                continue
            
            if self.index.get_postings(word) is None:
                continue
            
            # Calculate variant IDF
            df = len(self.index.get_postings(word))
            variant_idf = np.log((N - df + 0.5) / (df + 0.5))
            
            # Calculate variant TF
            # cwd = doc_word_counts.get(word)
            cwd = doc_word_counts[word]
            variant_tf_upper = (self.k1 + 1) * cwd
            variant_tf_lower = self.k1 * ((1 - self.b) + self.b * (doc_len / avg_doc_len)) + cwd
            variant_tf = variant_tf_upper / variant_tf_lower
            
            # Calculate normalized QTF
            # cwq = query_word_counts.get(word)
            cwq = query_word_counts[word]
            norm_qtf = (((self.k3 + 1) * cwq) / (self.k3 + cwq))
            
            score += variant_idf * variant_tf * norm_qtf

        # 4. Return score    
        return score


# TODO (HW4): Implement Personalized BM25
class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # TODO (HW4): Implement Personalized BM25
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid).get('length', 0)
        avg_doc_len = self.index.get_statistics()['mean_document_length']
        N = self.index.get_statistics()['number_of_documents']
        R = self.relevant_doc_index.get_statistics()['number_of_documents']
        score = 0
        
        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score   
        for word in query_word_counts:
            if word not in doc_word_counts.keys():
                continue
            
            if self.index.get_postings(word) is None:
                continue
            
            # Calculate personalized variant IDF
            df = len(self.index.get_postings(word))
            r = len(self.relevant_doc_index.get_postings(word)) if self.relevant_doc_index.get_postings(word) else 0
            personalized_idf_upper = (r + 0.5) * (N - df - R + r + 0.5)
            personalized_idf_lower = (df - r + 0.5) * (R - r + 0.5)
            personalized_idf = np.log(personalized_idf_upper / personalized_idf_lower)
            
            # Calculate variant TF
            # cwd = doc_word_counts.get(word, 0)
            cwd = doc_word_counts[word]
            variant_tf_upper = (self.k1 + 1) * cwd
            variant_tf_lower = self.k1 * ((1 - self.b) + self.b * (doc_len / avg_doc_len)) + cwd
            variant_tf = variant_tf_upper / variant_tf_lower
            
            # Calculate normalized QTF
            # cwq = query_word_counts.get(word, 0)
            cwq = query_word_counts[word]
            norm_qtf = (((self.k3 + 1) * cwq) / (self.k3 + cwq))
            
            score += personalized_idf * variant_tf * norm_qtf

        # 4. Return score    
        return score


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid).get('length', 0)
        # doc_len = self.index.get_doc_metadata(docid)['length']
        avg_doc_len = self.index.get_statistics()['mean_document_length']
        N = self.index.get_statistics()['number_of_documents']
        score = 0

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        for word in query_word_counts:
            if word not in doc_word_counts.keys():
                continue
            
            if self.index.get_postings(word) is None:
                continue
            
            # Calculate QTF
            # qtf = query_word_counts.get(word)
            qtf = query_word_counts[word]
            
            # Calculate normalized TF
            cwd = doc_word_counts.get(word)
            variant_tf_upper = 1 + np.log(1 + np.log(cwd))
            variant_tf_lower = 1 - self.b + self.b * (doc_len / avg_doc_len)
            variant_tf = variant_tf_upper / variant_tf_lower
            
            # Calculate IDF
            variant_idf = np.log((N + 1) / len(self.index.get_postings(word)))
            
            score += qtf * variant_tf * variant_idf

        # 4. Return the score
        return score


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid).get('length', 0)
        # doc_len = self.index.get_doc_metadata(docid)['length']
        score = 0

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF and IDF to get a score
        for word in query_word_counts:
            if word not in doc_word_counts.keys():
                continue
            
            if self.index.get_postings(word) is None:
                continue
            
            # Calculate TF
            tf = doc_word_counts.get(word)
            
            # Calculate IDF
            idf = 1 + np.log(len(self.index.document_metadata) / len(self.index.get_postings(word)))
            
            score += np.log(1 + tf) * idf

        # 4. Return the score
        return score


# TODO Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # doc_len = self.index.get_doc_metadata(docid).get('length', 0)
        N = self.index.get_statistics()['number_of_documents']
        score = 0
        
        for word in query_word_counts:
            if word not in doc_word_counts.keys():
                continue
            
            if self.index.get_postings(word) is None:
                continue
            
            # Calculate variant IDF
            df = len(self.index.get_postings(word))
            variant_idf = np.log((N - df + 0.5) / (df + 0.5))
            
            # Calculate TF
            tf = doc_word_counts.get(word)
            log_tf = np.log(1 + tf)
            
            score += variant_idf * log_tf
        
        return score


class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''
    def __init__(self, raw_text_dict: dict[int, str], 
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.raw_text_dict = raw_text_dict
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name)

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if docid not in self.raw_text_dict or not query:
            return 0

        # NOTE: unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        document_text = self.raw_text_dict[docid]
        score = self.cross_encoder_model.predict([(query, document_text)])[0]
        return score
