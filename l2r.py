from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer
import csv


# TODO: scorer has been replaced with ranker in initialization, check README for more details
class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.scorer = ranker
        self.feature_extractor = feature_extractor

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        # self.model = None # This should a LambdaMART object
        self.model = LambdaMART()
        pass
                   
    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        # TODO: for each query and the documents that have been rated for relevance to that query,
        for query, doc_relevance_lst in query_to_document_relevance_scores.items():
            query_parts = self.document_preprocessor.tokenize(query)
            
        # TODO: Accumulate the token counts for each document's title and content here
            doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
            title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)
            
        # TODO: For each of the documents, generate its features, then append
        # the features and relevance score to the lists to be returned
            for docid, relevance_score in doc_relevance_lst:
                features = self.feature_extractor.generate_features(docid, doc_word_counts, title_word_counts, query_parts)
                X.append(features)
                y.append(relevance_score * 2) # Double the training data's relevance score to avoid lightgbm error

        # TODO: Make sure to keep track of how many scores we have for this query in qrels
            num_docs_for_query = len(doc_relevance_lst)
            qgroups.append(num_docs_for_query)

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_term_counts = {}
        
        for term in query_parts:
            postings = index.get_postings(term)
            for docid, term_count in postings:
                if docid not in doc_term_counts:
                    doc_term_counts[docid] = {}
                doc_term_counts[docid][term] = term_count
        
        return doc_term_counts
        pass

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        query_to_document_relevance_scores = {}
        
        with open(training_data_filename, 'rt', encoding='cp850') as f:
            reader = csv.reader(f)
            # next(reader)
            
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                
                else:
                    query = row[header.index('query')]
                    docid = int(row[header.index('docid')])
                    relevance = float(row[header.index('rel')])
                    
                    if query not in query_to_document_relevance_scores:
                        query_to_document_relevance_scores[query] = []
                    
                    query_to_document_relevance_scores[query].append((docid, relevance))

        # TODO: prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures
        X, y, qgroups = self.prepare_training_data(query_to_document_relevance_scores)

        # TODO: Train the model
        self.model.fit(X, y, qgroups)
        pass

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # TODO: Return a prediction made using the LambdaMART model
        return self.model.predict(X)
        pass

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: the integer id of the user who is issuing the query or None if the user is unknown

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        
        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        #       pass these doc-term-counts to functions later, so we need the accumulated representations
        if query == "":
            return []
        query_parts = self.document_preprocessor.tokenize(query)
        if self.stopwords:
            query_parts = [token if token not in self.stopwords else None for token in query_parts]
        # query_word_counts = Counter(query_parts)
        
        doc_word_counts = defaultdict(Counter)
        possible_docs = set()
        for token in query_parts:
            postings = self.document_index.get_postings(token)
            if postings:
                for doc_id, term_freq in postings:
                    doc_word_counts[doc_id][token] = term_freq
                    possible_docs.add(doc_id)
                    
        if not possible_docs:
            return []

        # TODO: Accumulate the documents word frequencies for the title and the main body
        doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
        title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

        # TODO: Score and sort the documents by the provided scorer for just the document's main text (not the title).
        #       This ordering determines which documents we will try to *re-rank* using our L2R model
        # TODO: (HW4) support pseudofeedback arguments for the initial ranking
        # bm25 = self.scorer
        relevant_docs = self.scorer.query(query, pseudofeedback_num_docs, pseudofeedback_alpha, pseudofeedback_beta)
        
        # vector_ranker = self.scorer
        # relevant_docs = vector_ranker.query(query, pseudofeedback_num_docs, pseudofeedback_alpha, pseudofeedback_beta)
        top_100_docs = relevant_docs[:100]
        remaining_docs = relevant_docs[100:]

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        X_top_100 = []
        for top_doc_id, top_doc_score in top_100_docs:
            feature_vector = self.feature_extractor.generate_features(top_doc_id, doc_word_counts, title_word_counts, query_parts)
            X_top_100.append(feature_vector)

        # TODO: Use your L2R model to rank these top 100 documents
        ranked_top_100_scores = self.model.predict(X_top_100)
        ranked_top_100 = [(doc_id, score) for (doc_id, _), score in zip(top_100_docs, ranked_top_100_scores)]
        
        # TODO: Sort posting_lists based on scores
        ranked_top_100 = sorted(ranked_top_100, key=lambda x:x[1], reverse=True)
        
        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        remaining_docs = sorted(remaining_docs, key=lambda x:x[1], reverse=True)
        all_docs = ranked_top_100 + remaining_docs

        # TODO: Return the ranked documents
        return all_docs
        pass


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = list(recognized_categories)
        self.docid_to_network_features = docid_to_network_features
        self.ce_scorer = ce_scorer

        # TODO: For the recognized categories (i.e,. those that are going to be features), consider
        #       how you want to store them here for faster featurizing

        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid).get('length', 0)
        pass

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid).get('length', 0)
        pass

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        score = 0
        for word in query_parts:
            if word not in word_counts.keys():
                continue
            
            if index.get_postings(word) is None:
                continue
            
            score += np.log(1 + word_counts.get(word))
        
        return score
        pass

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        tfidf = TF_IDF(index)
        return tfidf.score(docid, word_counts, query_parts)
        pass

    # TODO: BM25
    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        bm25 = BM25(self.document_index)
        query_word_counts = Counter(query_parts)
        return bm25.score(docid, doc_word_counts, query_word_counts)
        pass

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        pivotNormalization = PivotedNormalization(self.document_index)
        query_word_counts = Counter(query_parts)
        return pivotNormalization.score(docid, doc_word_counts, query_word_counts)
        pass

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        binary_features = [0] * len(self.recognized_categories)
        document_categories = self.doc_category_info.get(docid, [])
        for category in document_categories:
            if category in self.recognized_categories:
                idx = self.recognized_categories.index(category)
                binary_features[idx] = 1
        return binary_features
        pass

    # TODO: PageRank
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        return self.docid_to_network_features.get(docid, {}).get('pagerank', 0)
        pass

    # TODO: HITS Hub
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        return self.docid_to_network_features.get(docid, {}).get('hub_score', 0)
        pass

    # TODO: HITS Authority
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        return self.docid_to_network_features.get(docid, {}).get('authority_score', 0)
        pass

    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        cross_encoder = self.ce_scorer
        return cross_encoder.score(docid, query)
        pass

    # TODO: Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str]) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        # NOTE: We can use this to get a stable ordering of features based on consistent insertion
        #       but it's probably faster to use a list to start

        feature_vector = []

        # TODO: Document Length
        # feature_vector.append(doc_word_counts[docid])
        feature_vector.append(self.get_article_length(docid))

        # TODO: Title Length
        # feature_vector.append(title_word_counts[docid])
        feature_vector.append(self.get_title_length(docid))

        # TODO Query Length
        feature_vector.append(len(query_parts))

        # TODO: TF (document)
        feature_vector.append(self.get_tf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF (title)
        feature_vector.append(self.get_tf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))

        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts))

        # TODO: Pagerank
        feature_vector.append(self.get_pagerank_score(docid))

        # TODO: HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))

        # TODO: HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))

        # TODO: (HW3) Cross-Encoder Score
        query_parts = [part for part in query_parts if part is not None]
        feature_vector.append(self.get_cross_encoder_score(docid, " ".join(query_parts)))

        # TODO: Add at least one new feature to be used with your L2R model.
        # title_query_overlap_count = 0
        # for title_word in title_word_counts:
        #     if title_word in query_parts:
        #         title_query_overlap_count += 1
        # feature_vector.append(title_query_overlap_count / len(query_parts))

        # TODO: Calculate the Document Categories features.
        for cate in self.get_document_categories(docid):
            feature_vector.append(cate)
        # NOTE: This should be a list of binary values indicating which categories are present.

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)
  
    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        # TODO: Generating the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)
        pass


