import asyncio
import glob
import json
import os
import time

import numpy as np

from .evaluators import Evaluator
from .baselines import ViLLMBaseModel

from asyncio import Semaphore
from datetime import datetime, timezone

from moviepy.editor import VideoFileClip, TextClip, ImageClip, CompositeVideoClip, concatenate_videoclips

import sys
sys.path.append('needlehastack/baselines')

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 model_to_test: ViLLMBaseModel = None,
                 evaluator: Evaluator = None,
                 needle = None,
                 needle_desc = None,
                 needle_modality = None,
                 needle_dir = "haystack",
                 haystack_dir = "haystack/ego",
                 retrieval_question = None,
                 results_version = 1,
                 context_lengths_min = 300,
                 context_lengths_max = 3000,
                 context_lengths_num_intervals = 100,
                 context_lengths = None,
                 video_depth_percent_min = 0,
                 video_depth_percent_max = 100,
                 video_depth_percent_intervals = 35,
                 video_depth_percents = None,
                 video_depth_percent_interval_type = "linear",
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 **kwargs):
        """
        :model_to_test: The model to test. Default is None.
        :evaluator: An evaluator to evaluate the model's response. Default is None.
        :param needle: The needle to be found in the haystack. Default is None.
        :param needle_desc: description of needle in other modality. Default is None.
        :param needle_modality: The modality of needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1 seconds.
        :param context_lengths_max: The maximum length of the context. Default is 320 seconds.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 40.
        :param context_lengths: The lengths of the context. Default is None.
        :param video_depth_percent_min: The minimum depth percent of the video. Default is 0.
        :param video_depth_percent_max: The maximum depth percent of the video. Default is 100.
        :param video_depth_percent_intervals: The number of intervals for the video depth percent. Default is 12.
        :param video_depth_percents: The depth percentages of the video. Default is None.
        :param video_depth_percent_interval_type: The type of interval for the video depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        :param kwargs: Additional arguments.
        """
        if not model_to_test:
            raise ValueError("A language model must be provided to test.")
        if not needle or not needle_modality or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, Needle_modality, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.needle_desc = needle_desc
        self.needle_modality = needle_modality
        self.needle_dir = needle_dir
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if video_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("video_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via video_depth_percent_intervals")

        if video_depth_percents is None:
            if video_depth_percent_min is None or video_depth_percent_max is None or video_depth_percent_intervals is None:
                raise ValueError("Either video_depth_percent_min, video_depth_percent_max, video_depth_percent_intervals need to be filled out OR the video_depth_percents needs to be supplied.")
            
            if video_depth_percent_interval_type == 'linear':
                self.video_depth_percents = np.round(np.linspace(video_depth_percent_min, video_depth_percent_max, num=video_depth_percent_intervals, endpoint=True)).astype(int)
            elif video_depth_percent_interval_type == 'sigmoid':
                self.video_depth_percents = [self.logistic(x) for x in np.linspace(video_depth_percent_min, video_depth_percent_max, video_depth_percent_intervals)]
            else:
                raise ValueError("video_depth_percent_interval_type must be either 'sigmoid' or 'linear' if video_depth_percents is None.")
        else:
            self.video_depth_percents = video_depth_percents
        

        self.model_to_test = model_to_test
        self.model_name = self.model_to_test.model_name
        
        self.evaluation_model = evaluator

    def logistic(self, x, L=100, x0=50, k=.1):
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.video_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.model_to_test.generate_prompt(self.retrieval_question)

        test_start_time = time.time()

        # Go see if the model can answer the question to pull out your random fact
        response = await self.model_to_test.generate(instruction=prompt, video_path=context)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        score = self.evaluation_model.evaluate_response(response)

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} seconds")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        model_file_location = self.model_name.replace(".", "_")
        context_file_location = f'{model_file_location}_modality_{self.needle_modality}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            context_dir = f"contexts/{model_file_location}"
            if not os.path.exists(context_dir):
                os.makedirs(context_dir)

            with open(f'{context_dir}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            result_dir = f"results/{model_file_location}"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # Save the result to file for retesting
            with open(f'{result_dir}/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def adjust_font_size(self, text, max_width, font='FreeSans', min_font_size=10, max_font_size=70):
        # Estimate the font size without creating a TextClip each time
        # This is a naive approach and may need adjustment based on the actual font metrics
        font_size = max_font_size
        # text_clip = TextClip(text, font_size=font_size, font=font)
        text_clip = TextClip(text, fontsize=font_size, font=font)
        text_width = text_clip.size[0]
        text_clip.close()
        font_size = int(font_size * max_width / text_width)
        return max(min(font_size, max_font_size), min_font_size)

    def insert_needle(self, context, depth_percent, context_length):
        
        video = VideoFileClip(context)
        
        # truncate video to context_length
        video_length = video.duration
        if video_length > context_length:
            video = video.subclip(0, context_length)

        # insert needle
        insertion_point = int(context_length * depth_percent / 100)
        new_context = context.replace('.mp4', f'_{self.needle_modality}_{context_length}_{insertion_point}.mp4')
        if os.path.exists(new_context): return new_context
        if self.needle_modality == 'text':
            # adjust text width
            max_text_width = video.size[0] * 0.9
            font_size = self.adjust_font_size(self.needle, max_text_width)
            # insert text needle
            text = TextClip(
                self.needle, 
                fontsize=font_size,
                # font_size=font_size, 
                color='white',
                bg_color='',
                font='FreeSans'
            ).set_position("bottom").set_duration(1).set_start(insertion_point)
            new_video = CompositeVideoClip([video, text])
        elif self.needle_modality == 'video':
            # insert video needle
            base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory
            self.needle = os.path.join(base_dir, self.needle_dir, self.needle)
            insert_video = VideoFileClip(self.needle).set_start(insertion_point).set_position("center")
            new_video = CompositeVideoClip([video, insert_video])
        elif self.needle_modality == 'image':
            # insert image needle
            base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory
            self.needle = os.path.join(base_dir, self.needle_dir, self.needle)
            insert_image = ImageClip(self.needle).set_duration(1).set_start(insertion_point).set_position("center")
            new_video = CompositeVideoClip([video, insert_image])
        else:
            video.close()
            raise NotImplementedError("Invalid needle modality, available needle: 'text', 'video', 'image")
        new_video.write_videofile(new_context, codec="libx264", fps=5, logger=None) # fps=video.fps
        # release resource
        video.close()
        if self.needle_modality == 'text': text.close()
        elif self.needle_modality == 'video': insert_video.close()
        elif self.needle_modality == 'image': insert_image.close()
        new_video.close()
        return new_context


    def get_context_length(self, context):
        with VideoFileClip(context) as video:
            return video.duration

    def read_context_files(self):
        
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        context_dir = os.path.join(base_dir, "context_videos")
        os.makedirs(context_dir, exist_ok=True)
        context = os.path.join(context_dir, f"{max_context_length}.mp4")
        if os.path.exists(context): return context # prevent repeating computing to save source

        print("start to load videos...")
        clips = []
        context_length = 0
        while context_length < max_context_length:
            # print("start to scan video")
            # print(os.path.join(base_dir, self.haystack_dir, "*.mp4"))
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.mp4")):
                # print(file)
                video = VideoFileClip(file)
                context_length += video.duration
                clips.append(video)
                if context_length >= max_context_length: break
        clip_video = concatenate_videoclips(clips)
        clip_video.write_videofile(context, codec="libx264", audio_codec="aac", logger=None)
        print("complete create context video.")

        for clip in clips:
            clip.close()
        clip_video.close()

        
        return context

    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Video Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Video Depths: {len(self.video_depth_percents)}, Min: {min(self.video_depth_percents)}%, Max: {max(self.video_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}: {self.needle_desc}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
