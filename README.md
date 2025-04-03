# Implicit Bias Injection

<img src=imgs/teaser.png  width="80%" height="60%">

This code is the official PyTorch implementation of [Implicit Bias Injection Attacks agasint Text-to-Image Diffusion Models](https://arxiv.org/abs/2411.03862).

If you have any questions, feel free to email <hyhuang@whu.edu.cn>

## Abstract
The proliferation of text-to-image diffusion models (T2I DMs) has led to an increased presence of AI-generated images in daily life. However, biased T2I models can generate content with specific tendencies, potentially influencing people’s perceptions. Intentional exploitation of these biases risks conveying misleading information to the public. Current research on bias primarily addresses explicit biases with recognizable visual patterns, such as skin color and gender. This paper introduces a novel form of implicit bias that lacks explicit visual features but can manifest in diverse ways across various semantic contexts. This subtle and versatile nature makes this bias challenging to detect, easy to propagate, and adaptable to a wide range of scenarios. We further propose an implicit bias injection attack framework (IBI-Attacks) against T2I diffusion models by precomputing a general bias direction in the prompt embedding space and
adaptively adjusting it based on different inputs. Our attack module can be seamlessly integrated into pre-trained diffusion models in a plug-and-play manner without direct manipulation of user input or model retraining. Extensive experiments validate the effectiveness of our scheme in introducing bias through subtle and diverse modifications while preserving the original semantics. The strong concealment and transferability of our attack across various scenarios further underscore the significance of our approach.

## Usage

### Training prompts generation
Use a LLM to generate a series of neutral and corresponding biased text prompts. The guiding prompts of the LLM can be found in the paper, which uses 200 prompt pairs for training.

### Generate direction vector and learn adaptive module
```
python learn_mapping.py \
    --model_path $path_to_sd2.1_diffusion_model \
    --json_file1 $path_to_the_JSON_file_containing_the_original_prompts \
    --json_file2 $path_to_the_JSON_file_containing_the_rephrased_prompts \
    --output_diff $path_to_save_the_computed_direction_vector \
    --output_model $path_to_save_the_trained_adaptive_module
```

### Perform bias injection
```
python run_bias_injection.py \
    --model_path $path_to_sd2.1_diffusion_model \
    --json_file $path_to_the_JSON_file_containing_the_evaluated_prompts \
    --se_model_path $path_to_trained_adaptive_module \
    --output_dir $path_to_save_the_generated_imgs \
    --bias_vector_path $path_to_diection_vector
```

### MLLM Evaluation
```
python llava_evaluation_large_scale.py \
    --model_id $path_to_llava-v1.6-mistral-7b-hf \
    --path1 $path_to_original_images_folder \
    --path2 $path_to_bias_injected_images_folder
```

## Cite
Welcome to cite our work if you find it is helpful to your research.
```
@inproceedings{huang2025implicit,
  title={Implicit Bias Injection Attacks against Text-to-Image Diffusion Models},
  author={Huang, Huayang and Jin, Xiangye and Miao, Jiaxu and Wu, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

```