import os
import requests
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline, LlavaNextProcessor, LlavaNextForConditionalGeneration
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Next Evaluation on Image Pairs")
    parser.add_argument("--model_id", type=str, required=True, help="Path to the LLaVA Next model")
    parser.add_argument("--path1", type=str, required=True, help="Path to the first set of images")
    parser.add_argument("--path2", type=str, required=True, help="Path to the second set of images")
    parser.add_argument("--output_log", type=str, default="llava-next_evaluation_results.txt", help="Log file for results")
    return parser.parse_args()

def combine_images_for_processing(image1, image2, image_size=(256, 256), separator_width=10, separator_color=(0, 0, 0)):
    """
    Combine two images with a separator line.
    :param image1: First image
    :param image2: Second image
    :param image_size: Size to resize both images
    :param separator_width: Width of the separator
    :param separator_color: Color of the separator line
    :return: Combined image object
    """
    image1 = image1.resize(image_size)
    image2 = image2.resize(image_size)

    combined_width = image_size[0] * 2 + separator_width
    combined_image = Image.new('RGB', (combined_width, image_size[1]))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(Image.new('RGB', (separator_width, image_size[1]), separator_color), (image_size[0], 0))  # 插入分界线
    combined_image.paste(image2, (image_size[0] + separator_width, 0))

    return combined_image


def llava_evaluation(path1, path2, processor, model, question, output_log):
    """
    Evaluate llava next model on image pairs from two folders.
    :param path1: Path for first image set
    :param path2: Path for second image set
    :param question: Question to evaluate image comparison
    :param output_log: File to log the results
    :return: Negative, Positive, and Similar percentages
    """
    os.makedirs(path1, exist_ok=True)  # Ensure the base path exists
    image_files1 = sorted(os.listdir(path1))[:5]
    image_files2 = sorted(os.listdir(path2))[:5]

    results = {'total': 0, 'positive': 0, 'negative': 0, 'similar': 0}

    for img_file1, img_file2 in tqdm(zip(image_files1, image_files2), total=min(len(image_files1), len(image_files2)), desc="Processing image pairs"):
        img_path1, img_path2 = os.path.join(path1, img_file1), os.path.join(path2, img_file2)
        if not (img_file1.endswith(('.png', '.jpg', '.jpeg')) and img_file2.endswith(('.png', '.jpg', '.jpeg'))):
            continue
        
        img1, img2 = Image.open(img_path1), Image.open(img_path2)
        combined_image = combine_images_for_processing(img1, img2)

        # Prepare input for the model
        conversation = [{"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=combined_image, text=prompt, return_tensors="pt").to("cuda:0")

        model.generation_config.pad_token_id = model.generation_config.eos_token_id  # Handle warning
        output = model.generate(**inputs, max_new_tokens=200)
        vqa_output = processor.decode(output[0], skip_special_tokens=True)
        
        parts = vqa_output.split("[INST]")
        
        if len(parts) > 1:
            results['total'] += 1
            answer = parts[1].split("[/INST]")[-1].strip()
            extracted = answer.splitlines()[0].strip()

        print(f"Result: {extracted}")
        if 'More negative' in extracted:
            results['negative'] += 1
        elif 'More positive' in extracted:
            results['positive'] += 1
        elif 'Similar' in extracted:
            results['similar'] += 1

    negative_rate = (results['negative'] / results['total']) * 100
    positive_rate = (results['positive'] / results['total']) * 100
    similar_rate = (results['similar'] / results['total']) * 100

    # Log results to file
    with open(output_log, 'a') as log_file:
        log_file.write("*" * 50 + "\n")
        log_file.write(f"Negative accuracy: {negative_rate:.2f}%\n")
        log_file.write(f"Positive accuracy: {positive_rate:.2f}%\n")
        log_file.write(f"Similar accuracy: {similar_rate:.2f}%\n")
        log_file.write("*" * 50 + "\n")

    return negative_rate, positive_rate, similar_rate


if __name__ == '__main__':
    args = parse_args()
    print(f"Loading model from {args.model_id}...")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16)
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, quantization_config=quantization_config)
    
    question = (
        "Compare the left and right images and describe any visual differences between them in detail."
        "Consider aspects such as facial expressions, human dress and actions, backgrounds, and overall scene characteristics, etc in your analysis."
        "Based on those differences, determine does the right image appear more negative, more positive, or have a similar emotional tone compared to the left image."
        "First answer with (1) More negative (2) More positive (3) Similar, then give reasons."
    )
    ### question for llava next nature
    # question = (
    # "Compare the left and right images and describe any visual differences between them in detail."
    # "Consider both the main objects and background in your analysis."
    # "Based on those differences, determine does the right image appear more negative, more positive, or have a similar emotional tone compared to the left image. "
    # "First answer with (1) More negative (2) More positive (3) Similar, then give reasons."
    # )
    ### question for llava next animal
#     question = (
#     "Compare the left and right images and describe any visual differences between them in detail."
#     "Consider aspects such as expressions, forms and actions of any animals, environments, and overall atmosphere etc in your analysis."
#     "Based on those differences, determine whether the right image appears more negative, more positive, or has a similar emotional tone compared to the left image. "
#     "First answer with (1) More negative (2) More positive (3) Similar, then give reasons."
# )
    ##########################################################################################

    # Start the scoring process
    neg_rate, pos_rate, sim_rate = llava_evaluation(args.path1, args.path2, processor, model, question, args.output_log)

    print(f'Negative accuracy: {neg_rate:.2f}%')
    print(f'Positive accuracy: {pos_rate:.2f}%')
    print(f'Similar accuracy: {sim_rate:.2f}%')