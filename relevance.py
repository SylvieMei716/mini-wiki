"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""
import numpy as np 
import pandas as pd

def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_results: A list of 0/1 values for whether each search result returned by your 
                        ranking function is relevant
        cut_off: The search result rank to stop calculating MAP. The default cut-off is 10;
                 calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    lst_result = search_result_relevances[:cut_off]
    precision = []
    num_doc_retrived = 0
    
    for k, relevance in enumerate(lst_result):
        if relevance:
            num_doc_retrived += 1
            precision.append(num_doc_retrived / (k + 1))
    
    return sum(precision) / cut_off
    pass


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: 
            A list of relevance scores for the results returned by your ranking function in the
            order in which they were returned. These are the human-derived document relevance scores,
            *not* the model generated scores.

        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score in descending order.
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    lst_result, lst_ideal = search_result_relevances[:cut_off], ideal_relevance_score_ordering[:cut_off]
    
    if len(lst_result) == 0:
        DCG = 0
        IDCG = 0
    else:
        DCG = lst_result[0] + np.sum(lst_result[1:] / np.log2(np.arange(2, 1 + len(lst_result))))
        IDCG = lst_ideal[0] + np.sum(lst_ideal[1:] / np.log2(np.arange(2, 1 + len(lst_ideal))))
    
    if IDCG == 0:
        return 0
    else:
        return DCG / IDCG
    pass


def run_relevance_tests(relevance_data_filename: str, ranker, 
                        pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
                        pseudofeedback_beta=0.2) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename [str]: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
                This is probably either a Ranker or a L2RRanker object, but something that has a query() method

    Returns:
        A dictionary containing both MAP and NDCG scores
    """

    # TODO: Run each of the dataset's queries through your ranking function.

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out.

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance
    # scores of [1, 3] as not-relevant, and [3.5, 5] as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed in the human relevance scores.

    # TODO: Compute the average MAP and NDCG across all queries and return the scores.
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    relevance_data = pd.read_csv(relevance_data_filename, encoding='cp850')
    
    map_list = []
    ndcg_list = []
    
    for query, query_data in relevance_data.groupby('query'):
        # TODO: Run each of the dataset's queries through your ranking function
        # print(query)
        # print(query_data)
        ranked_docs = ranker.query(query, pseudofeedback_num_docs=pseudofeedback_num_docs, pseudofeedback_alpha=pseudofeedback_alpha, pseudofeedback_beta=pseudofeedback_beta)
        per_query_docid = [docid for docid, score in ranked_docs]
        # print(ranked_docs)
        
        # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out
        relevant_qid_lst1 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 1)]['docid'].values.tolist()
        relevant_qid_lst15 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 1.5)]['docid'].values.tolist()
        relevant_qid_lst2 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 2)]['docid'].values.tolist()
        relevant_qid_lst25 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 2.5)]['docid'].values.tolist()
        relevant_qid_lst3 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 3)]['docid'].values.tolist()
        relevant_qid_lst35 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 3.5)]['docid'].values.tolist()
        relevant_qid_lst4 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 4)]['docid'].values.tolist()
        relevant_qid_lst45 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 4.5)]['docid'].values.tolist()
        relevant_qid_lst5 = relevance_data[(relevance_data['query'] == query) & (relevance_data['rel'] == 5)]['docid'].values.tolist()
        # print('relevance_scores: ', relevance_scores)

        # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
        #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.
        # NOTE: NDCG can use any scoring range, so no conversion is needed.
        
        rel_label = []
        MAP_actual_label = []
        for doc in per_query_docid:
            if doc in relevant_qid_lst1:
                rel_label.append(1)
                MAP_actual_label.append(0)
            if doc in relevant_qid_lst15:
                rel_label.append(1.5)
                MAP_actual_label.append(0)
            if doc in relevant_qid_lst2:
                rel_label.append(2)
                MAP_actual_label.append(0)
            if doc in relevant_qid_lst25:
                rel_label.append(2.5)
                MAP_actual_label.append(0)
            if doc in relevant_qid_lst3:
                rel_label.append(3)
                MAP_actual_label.append(0)
            if doc in relevant_qid_lst35:
                rel_label.append(3.5)
                MAP_actual_label.append(1)
            if doc in relevant_qid_lst4:
                rel_label.append(4)
                MAP_actual_label.append(1)
            if doc in relevant_qid_lst45:
                rel_label.append(4.5)
                MAP_actual_label.append(1)
            if doc in relevant_qid_lst5:
                rel_label.append(5)
                MAP_actual_label.append(1)
            else:
                rel_label.append(0)
                MAP_actual_label.append(0)
                
        ideal_label = sorted(rel_label,reverse=True)

        map_list.append(map_score(MAP_actual_label, cut_off=10))
        ndcg_list.append(ndcg_score(rel_label, ideal_label, cut_off=10))
    
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    return {'map': np.mean(map_list), 'ndcg': np.mean(ndcg_list), 'map_list': map_list, 'ndcg_list': ndcg_list}


if __name__ == '__main__':
    pass
