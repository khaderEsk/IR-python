import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    @staticmethod
    def retrieve(query_vec, index_model, documents):
        cosine_similarities = cosine_similarity(query_vec, index_model.getindex())

        results = pd.DataFrame(
            [
                {
                    "docno": index_model.getdocno(i),
                    "content": documents["text"].iloc[i],
                    "score": cosine_similarities[0][i],
                }
                for i in range(len(cosine_similarities[0]))
            ]
        )
        results = results.sort_values(by=["score"], ascending=False)
        results = results[results["score"] > 0]
        return results
