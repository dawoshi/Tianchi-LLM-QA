#!/usr/bin/env python
# coding: utf-8


from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from pdf_parse import DataProcess
import jieba

class BM25(object):

    def __init__(self, documents):

        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if(len(line)<5):
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            # docs.append(Document(page_content=tokens, metadata={"id": idx, "cate":words[1],"pageid":words[2]}))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            # full_docs.append(Document(page_content=words[0], metadata={"id": idx, "cate":words[1], "pageid":words[2]}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    # 初始化BM25的知识库
    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans

if __name__ == "__main__":

    # bm2.5
    dp =  DataProcess(pdf_path = "/root/autodl-tmp/codes/data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data
    bm25 = BM25(data)
    res = bm25.GetBM25TopK("座椅加热", 6)
    print(res)
