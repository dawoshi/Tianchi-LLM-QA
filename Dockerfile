# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04-cudnn

FROM registry.cn-shanghai.aliyuncs.com/aicar/vllm:base

# 如有安装其他软件的需求
# RUN apt-get update && apt-get install curl
# 如果安装其他python包的情况
#pip3 install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# RUN pip install --progress-bar off numpy pandas PyPDF2 langchain jieba rank_bm25 sentence-transformers faiss-gpu modelscope tiktoken transformers_stream_generator accelerate pdfplumber --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# 复制代码到镜像仓库
COPY app /app

# 指定工作目录
WORKDIR /app

# 容器启动运行命令
CMD ["bash", "run.sh"]
