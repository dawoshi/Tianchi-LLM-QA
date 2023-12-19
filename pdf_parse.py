#!/usr/bin/env python
# coding: utf-8

import pdfplumber
from PyPDF2 import PdfReader


class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = []
    def SlidingWindow(self, sentences, kernel = 512, stride = 1): 
        sz = len(sentences)
        cur = ""
        fast = 0
        slow = 0
        while(fast < len(sentences)):
            sentence = sentences[fast]
            if(len(cur + sentence) > kernel and (cur + sentence) not in self.data):
                self.data.append(cur + sentence + "。")
                cur = cur[len(sentences[slow] + "。"):]
                slow = slow + 1
            cur = cur + sentence + "。"
            fast = fast + 1
    
    def Datafilter(self, line, header, pageid, max_seq = 1024):

         sz = len(line)
         if(sz < 6):
             return

         if(sz > max_seq):

             if("■" in line):
                 sentences = line.split("■")
             elif("•" in line):
                 sentences = line.split("•")
             elif("\t" in line):
                 sentences = line.split("\t")
             else:
                 sentences = line.split("。")

             for subsentence in sentences:
                 subsentence = subsentence.replace("\n", "")

                 if(len(subsentence) < max_seq and len(subsentence) > 5):
                     # subsentence = subsentence.replace(",", "").replace("\n","").replace("\t","") + "\t" + header+ "\t" + str(pageid)
                     subsentence = subsentence.replace(",", "").replace("\n","").replace("\t","")
                     if(subsentence not in self.data):
                         self.data.append(subsentence)
         else:
             # line = line.replace("\n","").replace(",", "").replace("\t","") + "\t" +  header + "\t" + str(pageid)
             line = line.replace("\n","").replace(",", "").replace("\t","")
             if(line not in self.data):
                 self.data.append(line)
    # 提取页头即一级标题

    def GetHeader(self, page):
        try:
            lines = page.extract_words()[::]
        except:
            return None
        if(len(lines) > 0):
            for line in lines:
                if("目录" in line["text"] or ".........." in line["text"]):
                    return None
                if(line["top"] < 20 and line["top"] > 17):
                    return line["text"]
            return lines[0]["text"]
        return None

    # 按照每页中块提取内容,并和一级标题进行组合,配合Document 可进行意图识别
    def ParseBlock(self, max_seq = 1024):

        with pdfplumber.open(self.pdf_path) as pdf:

            for i, p in enumerate(pdf.pages):
                header = self.GetHeader(p)

                if(header == None):
                    continue

                texts = p.extract_words(use_text_flow=True, extra_attrs = ["size"])[::]

                squence = ""
                lastsize = 0

                for idx, line in enumerate(texts):
                    if(idx <1):
                        continue
                    if(idx == 1):
                        if(line["text"].isdigit()):
                            continue
                    cursize = line["size"]
                    text = line["text"]
                    # print(text, cursize)
                    if(idx == 0):
                        squence = squence + text
                        lastsize = cursize
                        continue
                    # print(cursize, text)
                    if(text == "□" or text == "•"):
                        continue
                    elif(text== "警告！" or text == "注意！" or text == "说明！"):
                        if(len(squence) > 0):
                            self.Datafilter(squence, header, i, max_seq = max_seq)
                            # print(squence, len(squence))
                        squence = ""
                    elif(format(lastsize,".5f") == format(cursize,".5f")):
                        if(len(squence)>0):
                            squence = squence + text
                        else:
                            squence = text
                    else:
                        lastsize = cursize
                        if(len(squence) < 15 and len(squence)>0):
                            squence = squence + text
                        else:
                            if(len(squence) > 0):
                                self.Datafilter(squence, header, i, max_seq = max_seq)
                                # print(squence, len(squence))
                            squence = text
                if(len(squence) > 0):
                    self.Datafilter(squence, header, i, max_seq = max_seq)
    def ParseOnePageWithRule(self, max_seq = 512, min_len = 6):
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if("...................." in text or "目录" in text):
                    continue
                if(len(text) < 1):
                    continue
                if(text.isdigit()):
                    continue
                page_content = page_content + text
            if(len(page_content) < min_len):
                continue
            if(len(page_content) < max_seq):
                if(page_content not in self.data):
                    self.data.append(page_content)
            else:
                sentences = page_content.split("。")
                cur = ""
                for idx, sentence in enumerate(sentences):
                    if(len(cur + sentence) > max_seq and (cur + sentence) not in self.data):
                        self.data.append(cur + sentence)
                        cur = sentence
                    else:
                        cur = cur + sentence
    #  滑窗法提取段落
    #  1. 把pdf看做一个整体,作为一个字符串
    #  2. 利用句号当做分隔符,切分成一个数组
    #  3. 利用滑窗法对数组进行滑动, 此处的
    def ParseAllPage(self, max_seq = 512, min_len = 6):
        all_content = ""
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if("...................." in text or "目录" in text):
                    continue
                if(len(text) < 1):
                    continue
                if(text.isdigit()):
                    continue
                page_content = page_content + text
            if(len(page_content) < min_len):
                continue
            all_content = all_content + page_content
        sentences = all_content.split("。")
        self.SlidingWindow(sentences, kernel = max_seq)

        # for idx, sentence in enumerate(sentences):
        #     if(len(cur + sentence) > max_seq and (cur + sentence) not in self.data):
        #         self.data.append(cur + sentence)
        #         cur = sentence
        #     else:
        #         cur = cur + sentence

if __name__ == "__main__":
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
    out = open("all_text.txt", "w")
    for line in data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
