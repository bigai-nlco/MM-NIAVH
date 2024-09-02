from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from jsonargparse import CLI

# from . import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from . import LLMNeedleHaystackTester
# from .evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator
from .evaluators import Evaluator, OpenAIEvaluator
from .baselines import ViLLMBaseModel
# from .providers import Anthropic, ModelProvider, OpenAI, Cohere

load_dotenv()

@dataclass
class CommandArgs():
    provider: str = "GPT4O"
    evaluator: str = "openai"
    evaluator_model_name: Optional[str] = "gpt-35-turbo-0125"
    # needle_modality: Optional[str] = "text"
    # needle: Optional[str] = "\nA young man is sitting on a piece of cloud in the sky, reading a book.\n"
    # needle_desc: Optional[str] = None
    # retrieval_question: Optional[str] = "What is the young man seated on a cloud in the sky doing?"
    retrieval_question: Optional[str] = "What is the young man seated on a cloud in the sky doing? Response the young man's action only."
    needle: Optional[str] = "readingsky.mp4" # 'readingsky.png', '\nA young man is sitting on a piece of cloud in the sky, reading a book.\n'
    needle_desc: Optional[str] = "the young man seated on a cloud in the sky is reading a book"
    # needle_desc: Optional[str] = "reading a book"
    needle_modality: Optional[str] = "video"
    needle_dir: Optional[str] = "haystack"
    haystack_dir: Optional[str] = "haystack/ego"
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1
    context_lengths_max: Optional[int] = 320
    context_lengths_num_intervals: Optional[int] = 40
    context_lengths: Optional[list[int]] = None
    video_depth_percent_min: Optional[int] = 0
    video_depth_percent_max: Optional[int] = 100
    video_depth_percent_intervals: Optional[int] = 12
    video_depth_percents: Optional[list[int]] = None
    video_depth_percent_interval_type: Optional[str] = "linear"
    num_concurrent_requests: Optional[int] = 1
    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = True
    final_context_length_buffer: Optional[int] = 200
    seconds_to_sleep_between_completions: Optional[float] = None
    print_ongoing_status: Optional[bool] = True
    # LangSmith parameters
    eval_set: Optional[str] = "multi-needle-eval-pizza-3"
    # Multi-needle parameters
    multi_needle: Optional[bool] = False # TODO: optimize multi_needle in video
    needles: list[str] = field(default_factory=lambda: [
        " Figs are one of the secret ingredients needed to build the perfect pizza. ", 
        " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ", 
        " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
    ])

def get_model_to_test(args: CommandArgs) -> ViLLMBaseModel:
    """
    Determines and returns the appropriate model provider based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        ModelProvider: An instance of the specified model provider class.
    
    Raises:
        ValueError: If the specified provider is not supported.
    """
    if args.provider.lower() == "gpt4o":
        from .baselines.gpt4o_modeling import GPT4O
        return GPT4O({"model_path": None, "device": 0})
    elif args.provider.lower() == "llavanext":
        from .baselines.llavanext_modeling import LLaVANeXT
        return LLaVANeXT({"model_path": "needlehaystack/baselines/checkpoints/LLaVA-NeXT-Video/LLaVA-NeXT-Video-7B-DPO", "device": 0})
    elif args.provider.lower() == "pllava":
        from .baselines.pllava_modeling import PLLaVA
        return PLLaVA({"model_path": "needlehaystack/baselines/checkpoints/PLLaVA/pllava-7b", "device": 0})
    elif args.provider.lower() == "malmm":
        from .baselines.malmm_modeling import MALMM
        return MALMM({"model_path": "needlehaystack/baselines/checkpoints/MALMM", "device": 0})
    elif args.provider.lower() == "longva":
        from .baselines.longva_modeling import LongVA
        return LongVA({"model_path": "needlehaystack/baselines/checkpoints/LongVA-7B", "device": 0})
    elif args.provider.lower() == "vila":
        from .baselines.vila_modeling import VILA
        return VILA({"model_path": "needlehaystack/baselines/checkpoints/Llama-3-LongVILA-8B-1024Frames", "device": 0})
    elif args.provider.lower() == "videollamb":
        from .baselines.videollamb_modeling import VideoLLaMB
        return VideoLLaMB({"model_path": "needlehaystack/baselines/checkpoints/videollamb-mem-llava-1.5-7b", "device": 0})
    else:
        raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        Evaluator: An instance of the specified evaluator class.
        
    Raises:
        ValueError: If the specified evaluator is not supported.
    """
    if args.evaluator.lower() == "openai":
        if args.needle_modality == "image" or args.needle_modality == "video":
            return OpenAIEvaluator(model_name=args.evaluator_model_name,
                                question_asked=args.retrieval_question,
                                true_answer=args.needle_desc)
        else:
            return OpenAIEvaluator(model_name=args.evaluator_model_name,
                                question_asked=args.retrieval_question,
                                true_answer=args.needle)
    elif args.evaluator.lower() == "langsmith":
        return LangSmithEvaluator()
    else:
        raise ValueError(f"Invalid evaluator: {args.evaluator}")

def main():
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)
    
    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else: 
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)
    tester.start_test()

if __name__ == "__main__":
    main()
