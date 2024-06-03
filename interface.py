from enum import Enum
from fastapi import FastAPI
from translate import translate_en, translate_default
import pandas as pd
from Preprocessor import Preprocessor
from Retriever import Retriever
from SearchEngine import SearchEngine
from langdetect import detect
app = FastAPI()
class DataName(str, Enum):
    antique = "antique"
    life_style = "life_style"
def text_pro(text, data_name):
    file_path = "antique.tsv"
    if data_name == "life_style":
        file_path = "lifestyle_dev.tsv"
    dataset = pd.read_csv(file_path, sep="\t")
    documents_df = pd.DataFrame(dataset)
    documents_df = documents_df[:100]
    ir_sys = SearchEngine(
        preprocessor=Preprocessor, retriever=Retriever, documents=documents_df
    )
    tran = translate_en(text)
    ss = ir_sys.querying(tran)
    result = ss[:10]["content"]
    end = []
    # if detect(text) == 'ar':
    for re in result:
        r = translate_default(str(re), text)
        end.append(r)
    return end
    # return result


@app.get("/data/{data_name}")
async def home(text: str, data_name: DataName):
    return {"data_name": data_name, "result": text_pro(text, data_name)}
