import asyncio
import glob
import json
import os
import time
from asyncio import Semaphore
from datetime import datetime, timezone

import numpy as np

from moviepy.editor import VideoFileClip, TextClip, ImageClip, CompositeVideoClip, concatenate_videoclips

from .evaluators import Evaluator
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .baselines import ViLLMBaseModel


class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    """
    Extends LLMNeedleHaystackTester to support testing with multiple needles in the haystack.
    
    Attributes:
        needles (list): A list of needles (facts) to insert into the haystack (context).
        model_to_test (ModelProvider): The model being tested.
        evaluator (Evaluator): The evaluator used to assess the model's performance.
        print_ongoing_status (bool): Flag to print ongoing status messages.
        eval_set (str): The evaluation set identifier.
    """
    def __init__(self, *args, 
                 needles=[], 
                 model_to_test: ViLLMBaseModel = None,
                 evaluator: Evaluator = None, 
                 print_ongoing_status = True,
                 eval_set = "multi-needle-eval-sf",
                 **kwargs):

        super().__init__(*args, model_to_test=model_to_test, **kwargs)
        self.needles = needles
        self.evaluator = evaluator
        self.model_to_test = model_to_test
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status
        self.insertion_percentages = []

    def adjust_font_size(self, text, max_width, font='Arial', min_font_size=10, max_font_size=70):
        # Estimate the font size without creating a TextClip each time
        # This is a naive approach and may need adjustment based on the actual font metrics
        font_size = max_font_size
        text_clip = TextClip(text, fontsize=font_size, font=font)
        text_width = text_clip.size[0]
        text_clip.close()
        font_size = int(font_size * max_width / text_width)
        return max(min(font_size, max_font_size), min_font_size)

    async def insert_needles(self, context, depth_percent, context_length):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context video at 
        designated depth percentages, effectively distributing these needles throughout the context. This method 
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of video 
        (haystack) based on the placement depth of these needles.
        
        Args:
            context (str): The original context video path.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens.
        
        Returns:
            str: The new context video with needles inserted.
        """
        video = VideoFileClip(context)
        
        # truncate video to context_length
        video_length = video.duration
        if video_length > context_length:
            video = video.subclip(0, context_length)
        
        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)
        
        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []

        # Insert needles at calculated points
        for needle in self.needles:


            insertion_point = int(context_length * depth_percent / 100)
            if self.needle_modality == 'text':
                # adjust text width
                max_text_width = video.size[0] * 0.9
                font_size = self.adjust_font_size(self.needle, max_text_width)
                # insert text needle
                text = TextClip(
                    self.needle, 
                    fontsize=font_size, 
                    color='white',
                    font='Arial'
                ).set_position("bottom").set_duration(1).set_start(insertion_point)
                video = CompositeVideoClip([video, text])
            elif self.needle_modality == 'video':
                # insert video needle
                insert_video = VideoFileClip(self.needle).set_start(insertion_point).set_position("center")
                video = CompositeVideoClip([video, insert_video])
            elif self.needle_modality == 'image':
                # insert image needle
                insert_image = ImageClip(self.needle).set_duration(1).set_start(insertion_point).set_position("center")
                video = CompositeVideoClip([video, insert_image])
            else:
                video.close()
                raise NotImplementedError("Invalid needle modality, available needle: 'text', 'video', 'image")
            # release resource
            if self.needle_modality == 'text': text.close()
            elif self.needle_modality == 'video': insert_video.close()
            elif self.needle_modality == 'image': insert_image.close()

            # Log 
            insertion_percentage = (insertion_point / context_length) * 100
            self.insertion_percentages.append(insertion_percentage)
            print(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")
            
            # Adjust depth for next needle
            depth_percent += depth_percent_interval  
        new_context = context.replace('.mp4', f'_{self.needle_modality}_{insertion_point}.mp4')
        video.write_videofile(new_context, codec="libx264", fps=5, logger=None) # fps=video.fps
        video.close()
        new_context = context
        return new_context



    async def generate_context(self, context_length, depth_percent):
        """
        Generates a context of a specified length and inserts needles at given depth percentages.
        
        Args:
            context_length (int): The total length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        
        Returns:
            str: The context with needles inserted.
        """
        context = self.read_context_files()
        context = await self.insert_needles(context, depth_percent, context_length)
        return context
    
    async def evaluate_and_log(self, context_length, depth_percent):
        """
        Evaluates the model's performance with the generated context and logs the results.
        
        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        """
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        test_start_time = time.time()

        # LangSmith
        ## TODO: Support for other evaluators 
        if self.evaluator.__class__.__name__ == "LangSmithEvaluator": 
            raise NotImplementedError("TODO: support langsmithevaluator") 
            print("EVALUATOR: LANGSMITH")
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluator.evaluate_chain(chain, context_length, depth_percent, self.model_to_test.model_name, self.eval_set, len(self.needles), self.needles, self.insertion_percentages)
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

        else:
            print("EVALUATOR: OpenAI Model")
            # Prepare your message to send to the model you're going to evaluate
            prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
            # Go see if the model can answer the question to pull out your random fact
            response = await self.model_to_test.generate(instruction=prompt, video_path=context)
            # Compare the reponse to the actual needle you placed
            score = self.evaluation_model.evaluate_response(response)

            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

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
                print (f"Context: {context_length} tokens")
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

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Video Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Video Depths: {len(self.video_depth_percents)}, Min: {min(self.video_depth_percents)}%, Max: {max(self.video_depth_percents)}%")
        print (f"- Needles: {self.needles}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
