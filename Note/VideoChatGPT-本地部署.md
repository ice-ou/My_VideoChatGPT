## 本地部署

在本地推理时，使用的是半精度模型，占用GPU显存约18GB。

### 下载示例视频

下载示例视频[下载链接](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/Ef0AGw50jt1OrclpYYUdTegBzejbvS2-FXBCoM3qPQexTQ?e=TdnRUG)
然后将其放在 `video_chatgpt/demo_sample_videos` 目录下。

### 运行Demo

```shell
python video_chatgpt/demo/video_demo.py 
        --model-name <path to the LLaVA-Lightening-7B-v1-1 weights prepared in step 3> \
        --projection_path <path to the downloaded video-chatgpt weights>
```
运行后打开网站即可访问。

**注意：**这里的LLaVA-Lightening-7B-v1-1权重和训练时的准备步骤相同，这里就不复述了。video-chatgpt的权重[下载链接](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1)，也可以使用自己训练的权重。