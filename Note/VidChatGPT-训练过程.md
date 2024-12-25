# VideoChatGPT 训练过程
我们在100K视频指令数据集([13303条视频](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/train_video_ids.txt))上训练Video ChatGPT模型。我们从LLaVA初始化训练。请按照以下说明训练Video-ChatGPT-7B型号。[参考链接](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/train_video_chatgpt.md)

## LLaVA 权重准备
VideoChatGPT是基于LLaVA构建的。所以需要获取LLaVA初始权重，其中包括已经微调后的7B大语言模型LLaMa。
- 首先，通过这里的[链接](https://huggingface.co/docs/transformers/main/model_doc/llama)下载LLaMA的原始权重。
- 然后使用以下脚本将LLaVA的Delta权重应用到原始LLaMA权重上，以获得LLaVA权重。
```shell
python scripts/apply_delta.py \ 
        --base-model-path <path to LLaMA 7B weights> \
        --target-model-path LLaVA-Lightning-7B-v1-1 \
        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
```
这里的Delta权重由模型微调得到，Delta权重通常只包含微调的增量变化，而不是完整的模型参数。  
上述命令将从HuggingFace上自动下载LLaVA-Lighting-7B-v1-1的Delta权重，并将其应用于提供的LLaMA权重，从而得到LLaVA-Lighting-7B-v1-1权重，并保存在当前目录中。

但是LLaMA初始权重获得比较麻烦，所以官方提供过了已经准备好的权重[链接](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1)。在下载时，需要下载所有文件，将其放在一个文件夹中。

## 训练数据集准备
**1. 下载100K视频指令数据集** 
[下载链接](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EWxYslvDeX1PijKWM_WxTkkBDXDDD350YnUQOkbcL8V7Xg?e=Lq9itD)。

**2. 将下载的json文件转换成训练格式**

```shell
python scripts/convert_instruction_json_to_training_format.py \
        --input_json_file <path to json file downloaded in step 2> \
        --output_json_file video_chatgpt_training.json
```
上述脚本会生成 `video_chatgpt_training.json` 文件用于训练。

**3. 下载 ActivityNet 数据集视频**

所有的视频数据都来源于ActivityNet dataset。我们提供其视频ID索引文件
[train_video_ids.txt](train_video_ids.txt)。 
可以通过官方提供的[下载链接](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EnLRDehrr8lGqHpC5w1zZ9QBnsiVffYy5vCv8Hl14deRcg?e=Ul5DUE)进行下载。

**4. 准备通过CLIP模型得到视频的时空特征**  
为了提高训练效率，这里预先计算了视频的时空特征，并在训练过程中直接使用它们。
下载视频数据后，可以通过以下脚本计算视频的时空特征。

```shell
python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path <path to the directory containing all the videos> \
        --clip_feat_path <The output dir to save the features in.>
```
该脚本将为每个视频生成时空特征
在`--clip_feat_path` 指定的目录中为每个视频保存一个pickle文件。  
也可以通过官方提供的[下载链接](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EnLRDehrr8lGqHpC5w1zZ9QBnsiVffYy5vCv8Hl14deRcg?e=Ul5DUE)，直接获取所有视频的时空特征文件。

## 训练VideoChatGPT模型

官方提供了在8块A100上进行分布式训练的示例脚本。
```shell
torchrun --nproc_per_node=8 --master_port 29001 video_chatgpt/train/train_mem.py \
          --model_name_or_path <path to LLaVA-7B-Lightening-v-1-1 model> \
          --version v1 \
          --data_path <path to the video_chatgpt using `convert_instruction_json_to_training_format.py` script.> \
          --video_folder <path to the spatio-temporal features generated in step 4 using `save_spatio_temporal_clip_features.py` script> \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
```

如果我们在单卡上进行训练，可以使用以下脚本：
```shell
python  video_chatgpt/train/train_mem.py \
          --model_name_or_path <path to LLaVA-7B-Lightening-v-1-1 model> \
          --version v1 \
          --data_path <path to the video_chatgpt using `convert_instruction_json_to_training_format.py` script.> \
          --video_folder <path to the spatio-temporal features generated in step 4 using `save_spatio_temporal_clip_features.py` script> \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
```
**注意：**使用单张4090显卡只能支持半精度训练：  
```python
model = VideoChatGPTLlamaForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float, # 使用半精度(fp16）或混合精度加载模型(bf16)
)
```