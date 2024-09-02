import sys
import os
import torch
import numpy as np
from lavis.models import load_model_and_preprocess

import decord
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge('torch')


from .base import ViLLMBaseModel



class MALMM(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        
        self.model_name = "MALMM"
        
        device = f'cuda:{model_args["device"]}'
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct_malmm", model_type="vicuna7b", is_eval=True, device=device, memory_bank_length=10, num_frames=20,
        )
        self.device = device

        
    async def generate(self, instruction, video_path):
        
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps
        
        video = load_video(vr, start_time=0, end_time=duration, fps=fps, num_frames=int(duration))
        video = self.vis_processors["eval"](video).to(self.device).unsqueeze(0)
        outputs = self.model.generate({"image": video, "prompt": f"Question: {instruction} Short answer:"})
        
        outputs = outputs[0].strip()
        # print(outputs)
        return outputs

def load_video(vr, start_time, end_time, fps, num_frames=20):
    start_index = int(round(start_time * fps))
    end_index = int(round(end_time * fps))
    select_frame_index = np.rint(np.linspace(start_index, end_index-1, num_frames)).astype(int).tolist()
    frames = vr.get_batch(select_frame_index).permute(3, 0, 1, 2).to(torch.float32)
    return frames