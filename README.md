# Classifier-Free Guidance inside the Attraction Basin May Cause Memorization
Official codebase for the paper "Classifier-Free Guidance inside the Attraction Basin May Cause Memorization"

## Getting Started

### Data
You can download the LAION-10K dataset from this [link](https://drive.google.com/drive/folders/1TT1x1yT2B-mZNXuQPg7gqAhxN_fWCD__?usp=sharing). 

### Pre-trained Models
You can download the finetuned SDv1.4 on 200 memorized samples from this [link](https://drive.google.com/drive/folders/1XiYtYySpTUmS_9OwojNo4rsPbkfCQKBl) and the training images from this [link](https://drive.google.com/drive/folders/1oQ49pO9gwwMNurxxVw7jwlqHswzj6Xbd). 

Sdv2.1 finetuned on the LAION-10K dataset is available on this [link](https://drive.google.com/file/d/1sNBcLASudpz09lvOghdMlKXTqwFLEh37/view?usp=share_link).

## Mitigating Memorization using Static Transition Point Method

```
$ python3 generate_test_dataset_stp.py --pretrained_model_name_or_path ./sd-21-finetuned_LAION10K/ --outdir LAION_10k_SDv21_0_500_7_5 --guidance_change_step 500 --guidance_scale 0.0 --guidance_scale_later 7.5
```

## Mitigating Memorization with Opposite Guidance and Static Transition Point Method

```
$ python3 generate_test_dataset_stp.py --pretrained_model_name_or_path ./sd-21-finetuned_LAION10K/ --outdir LAION_10k_SDv21_-2_650_7_5 --guidance_change_step 650 --guidance_scale -2.0 --guidance_scale_later 7.5
```


## Mitigating Memorization using Dynamic Transition Point Method

```
$ python3 generate_test_dataset_dtp.py 
```


## Mitigating Memorization using Opposite Guidance and Dynamic Transition Point Method

```
$ python3 generate_test_dataset_opposite_guidance.py 
```

