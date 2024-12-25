# VideoChatGPT-环境配置
项目Github地址：https://github.com/mbzuai-oryx/Video-ChatGPT

## 配置conda环境
```shell
conda create --name=video_chatgpt python=3.10
conda activate video_chatgpt

git clone https://github.com/mbzuai-oryx/Video-ChatGPT.git
cd Video-ChatGPT
pip install -r requirements.txt

# 将当前目录添加到 Python 的模块搜索路径中，使得 Python 可以找到并导入当前目录下的模块或包
export PYTHONPATH="./:$PYTHONPATH"
```
在环境安装过程注意cuda版本与torch版本的兼容。
## 在训练过程中还需要安装[FlashAttention](https://github.com/HazyResearch/flash-attention)
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout v1.0.7
python setup.py install
```
**注意**：在实际的训练过程中，显卡为A100或H100才使用FlashAttention，其他显卡(如4090)使用默认的Attention。在[*train_mem.py*](https://github.com/ice-ou/My_VideoChatGPT/blob/main/video_chatgpt/train/train_mem.py)文件中有体现。