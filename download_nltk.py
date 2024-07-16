import os
import socket
import socks
import nltk

proxy_address = ("127.0.0.1", 1080)
socks.set_default_proxy(socks.SOCKS5, *proxy_address, rdns=True)
socket.socket = socks.socksocket

# 指定下载目录
nltk_data_dir = os.path.expanduser("~/nltk_data")

# 创建目录（如果不存在）
os.makedirs(nltk_data_dir, exist_ok=True)

# 下载 cmudict 数据
nltk.download("cmudict", download_dir=nltk_data_dir)

# 加载 cmudict
from nltk.corpus import cmudict

pronouncing_dict = cmudict.dict()

# 示例：查找一个单词的发音
word = "example"
if word in pronouncing_dict:
    print(pronouncing_dict[word])
else:
    print(f"{word} not found in cmudict")
