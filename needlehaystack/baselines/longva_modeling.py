from .longva.model.builder import load_pretrained_model
from .longva.mm_utils import tokenizer_image_token, process_images
from .longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)

from .base import ViLLMBaseModel


class LongVA(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args["model_path"], model_args["device"])
        assert (
            "model_path" in model_args
            and "device" in model_args
        )
        self.model_name = "LongVA"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_args["model_path"], None, "llava_qwen", device_map="cuda:"+str(model_args["device"]))

    async def generate(self, instruction, video_path):
        
        gen_kwargs = {"do_sample": False, "use_cache": True, "max_new_tokens": 1024}
        
        preprompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        postprompt = "<|im_end|>\n<|im_start|>assistant\n"
        
        prompt = preprompt + "<image>" + instruction + postprompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_rate = vr.get_avg_fps()
        total_seconds = int(total_frames / frame_rate)
        uniform_sampled_frames = np.linspace(0, total_frames-1, total_seconds, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.device, dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return outputs



