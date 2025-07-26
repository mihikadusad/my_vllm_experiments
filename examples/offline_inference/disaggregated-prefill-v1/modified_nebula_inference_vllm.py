import copy
import os
import traceback
import uuid
import logging
import asyncio
import argparse
import json
import time
import numpy as np
from pathlib import Path
from vllm import SamplingParams
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from PIL import Image
import torch
from vllm.config import KVTransferConfig

# Import llava.mm_utils for tensor processing
try:
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
except ImportError:
    print("Warning: llava.mm_utils not available. Tensor processing will not work.")
    process_images = None


class NebulaVLLM():
    def __init__(self, MODEL_PATH, quantization=None, gpu_memory_utilization=0.9, pass_tensors=False):
        self.model_path = MODEL_PATH
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.pass_tensors = pass_tensors
        self.llm = None
        self.llava_image_processor = None
        self.llava_model_config = None
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,
            logprobs=None,
            top_p=1,
            top_k=1,
        )

    def get_ft_llava(self, model_path, model_base):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                               device_map='cuda')
        return tokenizer, model, image_processor, context_len

    async def initialize(self):

        # If pass_tensors is enabled, we need to get the image processor and model config
        if self.pass_tensors and process_images is not None:
            try:
                # Try to get the image processor and model config from the engine
                # This is a simplified approach - you might need to adjust based on your model setup
                # from transformers import AutoProcessor, AutoConfig
                # self.llava_image_processor = AutoProcessor.from_pretrained(self.model_path)
                # self.llava_model_config = AutoConfig.from_pretrained(self.model_path)
                # self.llava_model_config.tokenizer_padding_side = 'left'  # Use left padding for batch processing
                # self.llava_model_config.image_aspect_ratio = 'pad'  # TODO ablate different process methods
                # logging.info(f'Image process: {self.llava_model_config.image_aspect_ratio} ~~~~~')

                MODEL_PATH = '/workplace/data/face_efs/user/qzaamz/RekScienceOpenDetecto/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-0.5b-ov-20250321'
                self.llava_tokenizer, self.llava_model, self.llava_image_processor, _ = self.get_ft_llava(MODEL_PATH, None)
                self.llava_model.config.tokenizer_padding_side = 'left'  # Use left padding for batch processing

                self.llava_model.config.image_aspect_ratio = 'pad'  # TODO ablate different process methods
                logging.info(f'Image process: {self.llava_model.config.image_aspect_ratio} ~~~~~')
                self.llava_model_config = copy.deepcopy(self.llava_model.config)
                del self.llava_model
                del self.llava_tokenizer

                # if self.llava_tokenizer.pad_token is None:
                #     self.llava_tokenizer.pad_token = self.llava_tokenizer.eos_token

            except Exception as e:
                print(f"Warning: Could not load image processor or model config: {e}")
                print("Falling back to PIL image processing")
                self.pass_tensors = False

        """Async initialization of the model"""
        engine_args = AsyncEngineArgs(
    		model=self.model_path,
   	 	gpu_memory_utilization=self.gpu_memory_utilization,
    		dtype="auto",
    		quantization=self.quantization,
  	 	max_model_len=10000,
    		limit_mm_per_prompt={
        			"image": 1,
    		},
    		max_num_seqs=128,
    		enforce_eager=False,
   	    kv_transfer_config=KVTransferConfig(kv_connector="SharedStorageConnector",kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
            )
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)
        return self.llm

    async def generate_single(self, grounding_caption, image):
        """Async generation for single input"""
        prompt = f'<|im_start|>system You are a helpful assistant.<|im_end|><|im_start|>user Summarize this video.<|vision_start|><|image_pad|><|vision_end|><|im_end|><|im_start|>assistant'     
        # Process image based on pass_tensors setting
        if self.pass_tensors and process_images is not None and self.llava_image_processor is not None:
            try:
                # Convert PIL image to tensor using llava.mm_utils
                image_tensor = process_images([image], self.llava_image_processor, self.llava_model_config)
                # Use float16 for memory efficiency and speed
                image_tensor = [_image.half().cpu() for _image in image_tensor]
                
                # Extract single image from batch (remove batch dimension)
                processed_image = image_tensor[0].squeeze(0)
                
                # Normalize from [-1, 1] to [0, 1] range if needed to avoid PIL conversion error
                if processed_image.min() < 0:
                    processed_image = (processed_image + 1.0) / 2.0
                    processed_image = torch.clamp(processed_image, 0.0, 1.0)
                    processed_image = [processed_image]
                
            except Exception as e:
                print(f"Warning: Tensor processing failed, falling back to PIL image: {e}")
                # Fall back to PIL image processing
                processed_image = image
        else:
            # Use PIL image directly
            processed_image = image
        
        single_input = {
                "prompt": prompt,
            "multi_modal_data": {"image": processed_image},
        }

        # Generate unique request ID
        request_id = str(uuid.uuid4())
        generate_start_time = time.time()
        results_generator = self.llm.generate(single_input, self.sampling_params, request_id)
        
        model_output = None
        first_token_received = False
        ttft_latency = None
        
        async for request_output in results_generator:
            if not first_token_received:
                ttft_latency = time.time() - generate_start_time
                first_token_received = True
            model_output = request_output

        if model_output and model_output.outputs:
            return model_output.outputs[0].text.strip(), ttft_latency
        return "", ttft_latency

    async def close(self):
        """Clean up resources"""
        if self.llm:
            # AsyncLLMEngine doesn't have a direct close method, but we can set to None
            self.llm = None


def calculate_latency_metrics(latencies):
    """Calculate latency percentiles and statistics"""
    if not latencies:
        return {
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p90_latency": 0.0,
            "p99_latency": 0.0,
            "p100_latency": 0.0,
            "tm90_latency": None,
            "total_requests": 0
        }
    
    latencies = np.array(latencies)
    
    # Calculate percentiles
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    p100 = np.percentile(latencies, 100)
    
    # Calculate average
    avg_latency = np.mean(latencies)
    
    # Calculate tm90 (time to complete 90% of requests)
    # This is the time it takes to complete 90% of requests
    sorted_latencies = np.sort(latencies)
    tm90_idx = int(0.9 * len(sorted_latencies))
    tm90_latency = sorted_latencies[tm90_idx] if tm90_idx < len(sorted_latencies) else None
    
    return {
        "avg_latency": float(avg_latency),
        "p50_latency": float(p50),
        "p90_latency": float(p90),
        "p99_latency": float(p99),
        "p100_latency": float(p100),
        "tm90_latency": float(tm90_latency) if tm90_latency is not None else None,
        "total_requests": len(latencies)
    }


def load_image_text_pairs(text_file_path, image_dir_path):
    """Load image-text pairs from the specified format"""
    pairs = []
    
    with open(text_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # Split by tab character
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Line {line_num} has incorrect format: {line}")
                continue
                
            image_name, text = parts
            
            # Strip whitespace from both parts
            image_name = image_name.strip()
            text = text.strip()
            
            # Construct full image path
            image_path = os.path.join(image_dir_path, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                print(f"  Image name: '{image_name}' (length: {len(image_name)})")
                print(f"  Image dir: '{image_dir_path}'")
                print(f"  Full path: '{image_path}'")
                # Check if the image directory exists
                if not os.path.exists(image_dir_path):
                    print(f"  Error: Image directory does not exist: {image_dir_path}")
                else:
                    # List some files in the directory to help debug
                    try:
                        files_in_dir = os.listdir(image_dir_path)
                        print(f"  Files in directory ({len(files_in_dir)} total):")
                        for i, file in enumerate(files_in_dir[:5]):  # Show first 5 files
                            print(f"    {i+1}. '{file}'")
                        if len(files_in_dir) > 5:
                            print(f"    ... and {len(files_in_dir) - 5} more files")
                    except Exception as e:
                        print(f"  Error listing directory: {e}")
                continue
                
            pairs.append({
                'image_name': image_name,
                'image_path': image_path,
                'text': text
            })
    
    return pairs


async def process_image_text_pairs(model_path, text_file_path, image_dir_path, output_file_path, 
                                 quantization=None, gpu_memory_utilization=0.9, continuous_batch_size=1, max_files=None, pass_tensors=False):
    """Process all image-text pairs and save results"""
    
    # Load image-text pairs
    print(f"Loading image-text pairs from {text_file_path}")
    pairs = load_image_text_pairs(text_file_path, image_dir_path)
    print(f"Loaded {len(pairs)} image-text pairs")
    
    # Limit pairs if max_files is specified
    if max_files is not None:
        original_count = len(pairs)
        pairs = pairs[:max_files]
        print(f"Limited to {len(pairs)} pairs (from {original_count} total)")
    
    # Initialize model
    print(f"Initializing model: {model_path}")
    print(f"Pass tensors: {pass_tensors}")
    vlm = NebulaVLLM(model_path, quantization, gpu_memory_utilization, pass_tensors)
    await vlm.initialize()
    
    # Start timing after model initialization
    total_start_time = time.time()
    
    # Process pairs with concurrency control
    results = []
    latencies = []  # Track inference latencies
    total_pairs = len(pairs)
    
    print(f"\nStarting processing with maximum {continuous_batch_size} concurrent tasks")
    print(f"Total pairs to process: {total_pairs}")
    
    async def process_single_pair(pair, pair_index):
        nonlocal latencies
        print(f"Processing {pair_index + 1}/{total_pairs}: {pair['image_name']}")
        
        try:
            # Load image
            image = Image.open(pair['image_path']).convert('RGB')

            start_time_sample = time.time()

            # # Resize image to 384x384 while maintaining aspect ratio (fastest method)
            # target_size = (384, 384)
            # image.thumbnail(target_size, Image.Resampling.NEAREST)  # Fastest resampling method
            #
            # # Create a new 384x384 image with black background
            # resized_image = Image.new('RGB', target_size, (0, 0, 0))
            # # Paste the resized image centered
            # paste_x = (target_size[0] - image.size[0]) // 2
            # paste_y = (target_size[1] - image.size[1]) // 2
            # resized_image.paste(image, (paste_x, paste_y))
            # image = resized_image

            # Generate response
            response, ttft_latency = await vlm.generate_single(pair['text'], image)
            
            # Calculate inference latency
            inference_latency = time.time() - start_time_sample
            latencies.append(inference_latency)
            
            # Calculate image pixel count for correlation analysis
            image_width, image_height = image.size
            image_resolution = image_width * image_height

            result = {
                'image_name': pair['image_name'],
                'text': pair['text'],
                'response': response,
                'latency': inference_latency,
                'ttft_latency': ttft_latency,
                'image_pixels': image_resolution
            }
            
            # Print detailed information for latency correlation
            text_length = len(pair['text'])
            response_length = len(response)
            
            print(f"  Response: {response[:100]}...")
            print(f"  Latency: {inference_latency:.4f}s")
            print(f"  TTFT latency: {ttft_latency:.4f}s")
            print(f"  Image: {image_width}x{image_height} ({image_resolution:,} pixels)")
            print(f"  Text length: {text_length} chars")
            print(f"  Response length: {response_length} chars")
            
            return result
            
        except Exception as e:
            print(f"Error processing {pair['image_name']}: {e}")
            traceback.print_exc()
            return {
                'image_name': pair['image_name'],
                'text': pair['text'],
                'response': f"ERROR: {str(e)}",
                'latency': None,
                'ttft_latency': None,
                'image_pixels': None
            }
    
    async def run_pairs_concurrent(pairs_to_process, batch_size):
        active_tasks = set()
        collected_results = []
        
        for idx, pair in enumerate(pairs_to_process):
            if len(active_tasks) >= batch_size:
                done, active_tasks = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    result = await task
                    collected_results.append(result)
            
            task = asyncio.create_task(process_single_pair(pair, idx))
            active_tasks.add(task)
        
        # Wait for remaining tasks
        if active_tasks:
            remaining_results = await asyncio.gather(*active_tasks)
            collected_results.extend(remaining_results)
        
        return collected_results
    
    if continuous_batch_size == 1:
        # Process one by one (original behavior)
        for i, pair in enumerate(pairs):
            result = await process_single_pair(pair, i)
            results.append(result)
    else:
        # Process with controlled concurrency
        results = await run_pairs_concurrent(pairs, continuous_batch_size)
    
    # Calculate total processing time (excluding model initialization)
    total_processing_time = time.time() - total_start_time
    
    # Calculate throughput metrics
    throughput_images_per_second = len(results) / total_processing_time if total_processing_time > 0 else 0
    throughput_images_per_minute = throughput_images_per_second * 60
    
    # Calculate latency metrics
    latency_metrics = calculate_latency_metrics(latencies)
    
    # Print latency and throughput summary
    print(f"\nProcessing Summary:")
    print(f"  Total processing time (excluding init): {total_processing_time:.4f}s")
    print(f"  Total images processed: {len(results)}")
    print(f"  Throughput: {throughput_images_per_second:.2f} images/sec ({throughput_images_per_minute:.1f} images/min)")
    print(f"\nLatency Summary:")
    print(f"  Total requests: {latency_metrics['total_requests']}")
    print(f"  Average latency: {latency_metrics['avg_latency']:.4f}s")
    print(f"  P50 latency: {latency_metrics['p50_latency']:.4f}s")
    print(f"  P90 latency: {latency_metrics['p90_latency']:.4f}s")
    print(f"  P99 latency: {latency_metrics['p99_latency']:.4f}s")
    print(f"  P100 latency: {latency_metrics['p100_latency']:.4f}s")
    if latency_metrics['tm90_latency'] is not None:
        print(f"  TM90 latency: {latency_metrics['tm90_latency']:.4f}s")
    
    # Save results as JSON
    print(f"Saving results to {output_file_path}")
    
    # Create output structure
    output_data = {
        "model_info": {
            "model_path": model_path,
            "quantization": quantization,
            "gpu_memory_utilization": gpu_memory_utilization,
            "pass_tensors": pass_tensors
        },
        "processing_info": {
            "total_pairs": len(pairs),
            "processed_pairs": len(results),
            "text_file": text_file_path,
            "image_dir": image_dir_path,
            "continuous_batch_size": continuous_batch_size,
            "max_files": max_files,
            "total_processing_time": total_processing_time,
            "throughput_images_per_second": throughput_images_per_second,
            "throughput_images_per_minute": throughput_images_per_minute
        },
        "latency_metrics": latency_metrics,
        "results": results
    }
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {output_file_path}")
    
    # Cleanup
    await vlm.close()

# Script params
# --model_path /workplace/data/face_efs/user/linghanx/Nebula/weights/llava-onevision-qwen2-0.5b-ov-20250321-hf
# --text_file /workplace/data/face_efs/user/linghanx/Nebula/latency_test/data/fiona_testA_500_image_text_pairs.txt
# --image_dir /workplace/data/face_efs/user/linghanx/Nebula/latency_test/data/fiona_testA_500_image_text_pairs
# --output_file /workplace/data/motion_efs/home/girida/workplace/nebula/results/results_summary.json
# --continuous_batch_size 96 --max_files 500

def main():
    parser = argparse.ArgumentParser(description='Process image-text pairs with NebulaVLLM')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--text_file', type=str, required=True, help='Path to the text file with image-text pairs')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path for results (JSON format)')
    parser.add_argument('--quantization', type=str, default=None, help='Quantization method (e.g., awq)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    parser.add_argument('--continuous_batch_size', type=int, default=1, help='Maximum number of concurrent requests (default: 1)')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of image-text pairs to process (default: all)')
    parser.add_argument('--pass_tensors', action='store_true', help='Pass tensors for image processing (requires llava.mm_utils)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.text_file):
        print(f"Error: Text file not found: {args.text_file}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    if args.continuous_batch_size < 1:
        print(f"Error: continuous_batch_size must be >= 1, got {args.continuous_batch_size}")
        return
    
    if args.max_files is not None and args.max_files < 1:
        print(f"Error: max_files must be >= 1, got {args.max_files}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run processing
    asyncio.run(process_image_text_pairs(
        model_path=args.model_path,
        text_file_path=args.text_file,
        image_dir_path=args.image_dir,
        output_file_path=args.output_file,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        continuous_batch_size=args.continuous_batch_size,
        max_files=args.max_files,
        pass_tensors=args.pass_tensors
    ))


if __name__ == "__main__":
    main()
