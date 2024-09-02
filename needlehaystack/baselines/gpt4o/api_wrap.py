import os
import re
import json
import sys
import argparse
import openai
from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
from tqdm import tqdm
import time
import random
import base64
from mimetypes import guess_type
import cv2
from moviepy.editor import VideoFileClip

from os import getenv
from dotenv import load_dotenv
load_dotenv()
API_BASE = getenv("API_BASE")
API_KEY = getenv("API_KEY")


class BaseAPIWrapper(ABC):
    @abstractmethod
    def get_completion(self, user_prompt, system_prompt=None):
        pass

class OpenAIAPIWrapper(BaseAPIWrapper):
    def __init__(self, caller_name="default", api_base="https://api.openai.com",  key_pool=[], temperature=0, model="gpt-4o", time_out=5):
        if API_BASE != "":
            api_base = API_BASE
        key_pool = [API_KEY]
        print(api_base)
        print(key_pool)
        self.temperature = temperature
        self.model = model
        self.time_out = time_out
        self.api_key = random.choice(key_pool)
        self.api_base = api_base
        self.client = OpenAI(
            api_key=self.api_key,
            api_base=self.api_base,
        )

    def request(self, usr_question, system_content, video_path=None):
        
        base64Frames, audio_path = self.process_video(video_path, seconds_per_frame=1)
        # gpt4o limit
        base64Frames = base64Frames[-10:]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # {"role": "system", "content": f"{system_content}"},
                # {"role": "system", "content": "Use the video to answer the provided question."},
                {"role": "user", "content": [
                    {"type": "text", "text": "These are the frames from the video."},
                    *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                    {"type": "text", "text": usr_question},
                    
                ],}
            ],
            temperature=0,
        )


        # resp = response.choices[0]['message']['content']
        # total_tokens = response.usage['total_tokens']
        resp = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        

        return resp, total_tokens
    
    def get_completion(self, user_prompt=None, system_prompt=None, video_path=None, max_try=10):
        gpt_cv_nlp = ""
        key_i = 0
        total_tokens = 0
        # TODO: improve naive cache
        cached = 0
        cache_dir = "cache_dir/gpt4o_cache"
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        if video_path is not None:
            cache_path = cache_dir + "/" +user_prompt.replace(" ","").replace("/","").replace(".","").replace("'","").replace("\n","")+video_path.replace("/","").replace(".","")+".json"
            if os.path.exists(cache_path):
                cache = json.load(open(cache_path))
                gpt_cv_nlp = cache["response"]
                total_tokens = cache["tokens"]
                cached = 1
        gpt_cv_nlp, total_tokens = self.request(user_prompt, system_prompt, video_path)
        res = {
            "response": gpt_cv_nlp,
            "tokens": total_tokens
        }
        # while max_try > 0 and cached == 0:
        #     try:
        #         gpt_cv_nlp, total_tokens = self.request(user_prompt, system_prompt, video_path)
        #         res = {
        #             "response": gpt_cv_nlp,
        #             "tokens": total_tokens
        #         }
        #         # if video_path is not None: # FIXME: file name too long
        #         #     cache_path = cache_dir+"/"+user_prompt.replace(" ","").replace("/","").replace(".","").replace("'","").replace("\n","")+video_path.replace("/","").replace(".","")+".json"        
        #         #     json.dump(res, open(cache_path, "w"))
        #         max_try = 0
        #         break
        #     except Exception as e:
        #         # if e.code == "content_filter":
        #         #     gpt_cv_nlp, total_tokens = "", 0
        #         #     break
        #         print(f"encounter error: {e}")
        #         print("fail ", max_try)
        #         # key = self.key_pool[key_i%2]
        #         # openai.api_key = key
        #         # key_i += 1
        #         time.sleep(self.time_out)
        #         max_try -= 1
    
        return gpt_cv_nlp, total_tokens



    def process_video(self, video_path, seconds_per_frame=2):
        base64Frames = []
        base_video_path, _ = os.path.splitext(video_path)

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame=0

        # Loop through the video and extract frames at specified sampling rate
        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip
        video.release()

        # # Extract audio from video
        audio_path = f"{base_video_path}.mp3"
        # clip = VideoFileClip(video_path)
        # clip.audio.write_audiofile(audio_path, bitrate="32k")
        # clip.audio.close()
        # clip.close()

        # print(f"Extracted {len(base64Frames)} frames")
        # print(f"Extracted audio to {audio_path}")
        # print(base64Frames)
        return base64Frames, audio_path
