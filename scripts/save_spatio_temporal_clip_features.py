import os
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor


def load_video(vis_path, num_frm=100):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)  # 实际要处理的视频帧数
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)  # 获取实际处理视频帧的索引列表
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3),这里n_clips*num_frm 可能是为了考虑未来可能支持多剪辑的情况

    # Resize image to 224x224
    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    # （T,H,W,3）->(1,T,H,W,3))
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    # 将视频平均分成desired_num_frames份，每份的帧数为seg_size
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        # 取每份的中间帧作为当前帧
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape # (T, 256, 1024) 256=16x16

    temporal_tokens = np.mean(features, axis=1)  # 空间平均池化 ，得到时间特征 [T, 1024]
    padding_size = num_temporal_tokens - t # 这里是100-T，即需要补齐的帧数（处理输入的视频的帧数不足100的情况）
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0) # 时间平均池化，得到空间特征 [256, 1024]
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0) # 将时空特征拼接，得到 [T+256, 1024] T=100

    return sp_features


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=32,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args


''' 使用CLIP预先提取提取视频特征，并保存到本地，方便后续使用。'''
def main():
    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower.eval()

    all_videos = os.listdir(video_dir_path)
    video_clip_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
            continue
        try:
            video = load_video(video_path) # 导入视频 [1, T, H, W, 3]
            video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values'] # [1, T, 3, H, W] 每个像素的值
            video_tensor = video_tensor.half() # Convert to half precision float16

            n_chunk = len(video_tensor) # 视频帧数 T
            video_features = torch.FloatTensor(n_chunk, 256, 1024).fill_(0) # 初始化全零张量 [T, 256, 1024] 每一帧的特征向量
            n_iter = int(math.ceil(n_chunk / float(infer_batch))) # 迭代次数
            for i in range(n_iter):
                min_ind = i * infer_batch  # 起始帧索引
                max_ind = (i + 1) * infer_batch  # 结束帧索引
                video_batch = video_tensor[min_ind:max_ind].cuda() # 提取当前batch的帧数据，并转移到GPU上进行加速计算

                image_forward_outs = vision_tower(video_batch, output_hidden_states=True) # 使用CLIP模型进行特征提取

                select_hidden_state_layer = -2
                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]  # 提取倒数第二层隐藏层作为特征向量
                batch_features = select_hidden_state[:, 1:]  # 去掉第一个CLIP的CLS token [batch_size, 256, 1024]
                video_features[min_ind:max_ind] = batch_features.detach().cpu() # 转移到cpu，并使用detach() 方法，防止梯度计算

            video_clip_features[video_id] = get_spatio_temporal_features(video_features.numpy().astype("float16")) # 将特征向量保存到字典中
            counter += 1

        except Exception as e:
            print(f"Can't process {video_path}")
        
        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            for key in video_clip_features.keys():
                features = video_clip_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:  # 将特征向量[T+256,1024]保存到本地pkl文件中
                    pickle.dump(features, f)
            video_clip_features = {}
    
    # 最后保存剩余的视频
    for key in video_clip_features.keys():
        features = video_clip_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()
