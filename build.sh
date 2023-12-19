#!/bin/bash

# 以下命令作为打包示例，实际使用时请修改为自己的镜像地址, 建议每次提交前完成版本修改重新打包
# docker build -t registry.cn-shanghai.aliyuncs.com/taylor:0.1 .

imageid=`docker images|awk 'NR>1'|grep "aicar/taylor"|awk '{print($3)}'`
echo $imageid
docker rmi -f $imageid
echo yes|docker builder prune
docker build -t registry.cn-shanghai.aliyuncs.com/taylor:0.1 .

ImageId=`docker images|awk 'NR>1'|grep "0.1"|awk '{print($3)}'`
echo $ImageId
docker tag $ImageId registry.cn-shanghai.aliyuncs.com/aicar/taylor:v1
docker login --username=xxx -p xxx registry.cn-shanghai.aliyuncs.com
docker push registry.cn-shanghai.aliyuncs.com/aicar/taylor:v1



