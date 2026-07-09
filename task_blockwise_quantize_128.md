阅读以下要求，帮我列一个详细的实验计划，把计划反复review三遍直到没有问题以后，写入PLAN_blockwise_quantize_128.md。当前 W8A8 MegaMoE 的量化粒度是 group 32 的，我现在想让 W8A8 MegaMoE 支持 blockwise 128 量化（通过一个环境变量控制），我目前想到的一个方案是仍然按照 blockwise 128 粒度进行量化，但是在 m 维度 repeat_interleave 128 次，n 维度 repeat_interleave 4 次，这样 scale 的 shape 仍然是 group 32 的 scale 的 shape，只是之前是 32 个元素共用一个 scale，现在 128*128 个元素共用一个 scale，仍然可以在当前 MegaMoE 实现下进行 block_scaled 32 的 mma 运算，MegaMoE kernel 部分需要处理的应该就是 epilogue warp 的量化部分，将其按照我上述逻辑进行操作即可。

请你根据我上述描述的思路，根据当前 W8A8 MegaMoE 的代码实现，评估一下我的实现方案的可行性、不足之处，有更好的方案的话提出更好的方案，根据你所评估选出来的方案列出详细的实现方案（包括需要改动哪些代码，对性能是否有影响多大影响，对精度是否有影响等等），把计划反复review三遍直到没有问题以后，写入PLAN_blockwise_quantize_128.md。

# 实验可用的GPU信息
## 机器列表
本机的后四台显卡都可以使用

## 机器配置
* 每台机器的配置都是 8xB200(180G显存)

# 代理配置
* 如果需要通过网络下载第三方库，请运行下面命令使用代理：
```
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com,0.0.0.0,baidu-int.com,aliyun.com,127.0.0.1,.baidu.com,.bcebos.com 
export http_proxy=http://agent.baidu.com:8891 
export https_proxy=http://agent.baidu.com:8891
```