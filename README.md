# AR Environment Design with Stable Diffusion

This repository contains the server and benchmarking implementation for the project **_AR Environment Design with Stable Diffusion_**


## Setup
To use this repository, you can generate a Conda environment using `environment.yml` by running
```sh
conda env create -f environment.yml  --name <custom_name>
```

## Benchmark Run
Before running benchmarking, download the the preprocessed and labelled subset of NYUv2 depth dataset from [this link](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and place the dataset into the **data** folder
Once the data is ready, benchmarking can be run by running
```sh
python .\models\controlnet_model_wrapper.py --num_steps <default:10> --resolution <default:512> --condition_type <default:depth, options: depth, seg> --multi_condition <defalt: False>
```
Example:
```sh
python .\models\controlnet_model_wrapper.py --num_steps 10 --resolution 512 --condition_type seg
```

The following argument options are available:
- **data_path**      : set a custom path to retrieve benchmark dataset from
- **result_root**    : set a custom path to save results to - the folder structure of the results folder has to replicated for this to work 
- **prompt**         : set a custom prompt to apply to all images - when left as default, a random prompt is sampled from the set of prompts retrieved from Text2Room and SceneScape papers.
- **guidance_scale** : set the guidance scale parameter (0-10) of the ControlNet model, default: 7.5
- **num_steps**      : set the number of inference steps to run ControlNet for, default: 10
- **condition_type** : choose a condition type among "seg" (segmentation) and "depth" 
- **multi_condition**: use both depth and segmentation conditioning - overrides the **condition_type** argument
- **cache_dir**      : to use a custom directory to cache model files to - may be useful when e.g you are using a cluster and there is a low storage limit in your home directory but high limit in your scratch folder
- **display **       : whether to display generated depth maps, segmentation maps, RGB images as the model runs
- **resolution**     : the resolution to downsample the image to before being used for generation - the output is resized to the original image size


## Deploying the server
The server can be deployed for client applications to connect to by running
```sh
python .\server\server_app.py --num_steps <default:10> --resolution <default:512> --condition_type <default:depth, options: depth, seg>
```
Example:
```sh
python .\server\server_app.py --num_steps 20 --resolution 384 --condition_type seg
```

The argument set of the server is more restricted than benchmarking, allowing for the following options:
- cache_dir      : to use a custom directory to cache model files to - may be useful when e.g you are using a cluster and there is a low storage limit in your home directory but high limit in your scratch folder
- resolution     : the resolution to downsample the image to before being used for generation - the output is resized to the original image size
- num_steps      : set the number of inference stpes to run ControlNet for, default: 10
- condition_type : choose a condition type among "seg" (segmentation) and "depth" 
- multi_condition: use both depth and segmentation conditioning - overrides the "condition_type" argument

Once the server is deployed and its IP address is displayed in the terminal, use this address on the client device to connect to the server.

## Repository Structure
The folders of this repository contain the following sections of the pipeline:
- **models:** The main generation pipeline script (controlnet_model_wrapper.py) and relevant model data.
- **models_3d:** Functions for creating and processing point clouds and meshes.
- **server:** The Flask server script and the user input handler (arcore_handler.py) that runs the generation pipeline according to the request received by the server.
The output files and evaluation results for generation requests made by the client app are placed in server/user_data folder
- **utils:** Result evaluation class (evaluation.py), input preprocessing functions (preprocessing.py) and result visualisation functions (visualisation.py)
- **data:** Where the preprocessed and labelled subset of NYUv2 depth dataset file is placed when carrying out benchmarking
- **results:** Where the output files and evaluation results are saved.

Both the results folder and the server/user_data folder are structured as follows:
- Received original RGB images and depth maps are directly stored at folder level.
- **ControlNet**
    - **2d_images:** generated RGB images and their comparisons with ground truth RGB images
    - **depth_map_heatmaps:** heatmaps comparing ground truth, predicted ground truth and predicted generated depth maps
    - **eval_logs:** evaluation results
    - **gen_depth_maps:** depth maps for the generated RGB images
    - **gen_point_clouds:** point clouds for the generated RGB images
    - **ground_conditions:** inferred segmentation maps are stored here
    - **predicted_ground_truth_depth_maps:** depth maps inferred on the gorund truth RGB image are stored here
- **ground_point_clouds:** point clouds for the ground truth RGB image 
- **processed_user_data:** final generated mesh files

## Further Information
Please refer to the project report _AR_Environment_Design_with_Stable_Diffusion_Semester_Project_Report.pdf_

## Authors
Mert Ertugrul 
