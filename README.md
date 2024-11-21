# Classifier-Free Guidance inside the Attraction Basin May Cause Memorization
Official codebase for the paper "Classifier-Free Guidance inside the Attraction Basin May Cause Memorization"

## Getting Started

You can download the LAION-10K dataset from this link 



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


## Mitigating Memorization using Opposite Guidance and Static Transition Point Method


```
$ python3 generate_test_dataset_dtp.py 
```

