
def recall_at_k(ground_truth: list[str], predicted: list[str], k: int = 0) -> float:
    """
    This metric gives how many actual relevant results were shown out of all actual relevant results.

    Args:
        ground_truth (list[str]): pmids list of the relevant documents
        predicted (list[str]): list of pmids returned by the retriever
        k (int, optional): top k items to consider. Defaults to len(predicted).

    Returns:
        float: recall@k
    """
    if k == 0:
        k = len(predicted)
    act_set = set(ground_truth)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 4)
    return result


def precision_at_k(ground_truth: list[str], predicted: list[str], k: int = 0) -> float:
    """
    This metric quantifies how many items in the top-K results were relevant.

    Args:
        ground_truth (list[str]): pmids list of the relevant documents
        predicted (list[str]): list of pmids returned by the retriever
        k (int, optional): top k items to consider. Defaults to len(predicted).

    Returns:
        float: precision@k
    """
    if len(predicted) != 0:
        if k == 0:
            k = len(predicted)
        act_set = set(ground_truth)
        pred_set = set(predicted[:k])
        result = round(len(act_set & pred_set) / float(k), 4)
    else:
        result = 0.0
    return result


def f1_at_k(ground_truth: list[str], predicted: list[str], k: int = 0) -> float:
    """
    This is a combined metric that incorporates both Precision@k and Recall@k by taking their harmonic mean.

    Args:
        ground_truth (list[str]): pmids list of the relevant documents
        predicted (list[str]): list of pmids returned by the retriever
        k (int, optional): top k items to consider. Defaults to len(predicted).

    Returns:
        float: F1@k
    """
    if k == 0:
        k = len(predicted)
    recall = recall_at_k(ground_truth, predicted, k)
    precision = precision_at_k(ground_truth, predicted, k)
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = (2 * recall * precision)/(recall + precision)
    return round(f1, 4)
    

"""
predicted = ["6"]
ground_truth = ["1", "3", "5"]
print(f"Precision = {precision_at_k(ground_truth, predicted)}")
print(f"Recall = {recall_at_k(ground_truth, predicted)}")
print(f"F1 = {f1_at_k(ground_truth, predicted)}\n")
for k in range(1, 6):
    print(f"Precision@{k} = {precision_at_k(ground_truth, predicted, k)}")
    print(f"Recall@{k} = {recall_at_k(ground_truth, predicted, k)}")
    print(f"F1@{k} = {f1_at_k(ground_truth, predicted, k)}\n")
"""