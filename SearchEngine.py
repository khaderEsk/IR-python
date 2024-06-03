import pandas as pd
from IndexModel import IndexModel


class SearchEngine:
    def __init__(self, preprocessor, retriever, documents):
        self.documents = documents
        self.preprocessor = preprocessor
        self.retriever = retriever
        self.model = None
        self.rebuild()

    def rebuild(self):
        self.documents["processed_text"] = self.documents["text"].apply(
            self.preprocessor.process
        )
        print("Prepocessing Finished")
        self.model = IndexModel(self.documents)

    def querying(self, query):
        query_prc = self.preprocessor.process(query)
        query_vec = self.model.vectorize(query_prc)
        docs_res = self.retriever.retrieve(query_vec, self.model, self.documents)
        return docs_res

    def transform(self, input_df):
        combined_results = []
        for _, row in input_df.iterrows():
            res_on_query = self.querying(row["query"])
            res_on_query["qid"] = row["qid"]
            res_on_query["rank"] = list(range(res_on_query.shape[0]))
            combined_results.append(res_on_query)
        transformed_df = pd.concat(combined_results, ignore_index=True)
        return transformed_df

