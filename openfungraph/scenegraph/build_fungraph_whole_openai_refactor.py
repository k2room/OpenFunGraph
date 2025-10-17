import gc
import gzip
import json
import os
import pickle as pkl
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Union
from textwrap import wrap
from openfungraph.utils.general_utils import prjson
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm, trange
from transformers import logging as hf_logging
from openfungraph.llava.llava_model_16 import LlavaModel16
from openfungraph.slam.slam_classes import MapObjectList
from openfungraph.scenegraph.GPTPrompt import GPTPrompt


torch.autograd.set_grad_enabled(False)
hf_logging.set_verbosity_error()

# Import OpenAI API
import openai
from openai import OpenAI
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

_INTERACTABLE_MASK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mask":   {"type": "array", "items": {"type": "boolean"}},
        "reason": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["mask", "reason"]
}

# --- Responses API helpers (role-aware) ---
def _to_responses_input(msgs):
    """
    [{role, content(str)}] -> Responses API 'input' 형식으로 변환.
    - user/system -> type='input_text'
    - assistant   -> type='output_text'
    """
    out = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        part_type = "output_text" if role == "assistant" else "input_text"
        out.append({
            "role": role,
            "content": [{"type": part_type, "text": content}],
        })
    return out

def _as_responses_input(payload):
    """
    list(messages) 또는 str/dict를 Responses API 'input'으로 포장
    """
    if isinstance(payload, list):
        return _to_responses_input(payload)
    if isinstance(payload, dict):
        s = json.dumps(payload, ensure_ascii=False)
    else:
        s = str(payload)
    return [{"role": "user", "content": [{"type": "input_text", "text": s}]}]

def _resp_text(resp):
    """Responses 응답에서 텍스트 안전 추출"""
    try:
        if getattr(resp, "output_text", None):
            return resp.output_text.strip()
    except Exception:
        pass
    try:
        return resp.output[0].content[0].text.strip()
    except Exception:
        return ""

def _ask_gpt4_text(payload, timeout=60):
    """payload: list(messages) 또는 str/dict"""
    resp = client.responses.create(
        model="gpt-4",
        input=_as_responses_input(payload), 
        timeout=timeout,
    )
    return _resp_text(resp)

def _ask_json_response(payload, timeout=60, model="gpt-4o-mini"):
    """
    Responses API로 '반드시 JSON(스키마 강제)'을 돌려받는다.
    payload: list(messages) 또는 str/dict (당신의 _as_responses_input 으로 감쌈)
    """
    resp = client.responses.create(
        model=model,  # ← "gpt-4" 대신 최신 모델
        input=_as_responses_input(payload),
        # JSON 스키마 강제
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "InteractableMask",
                "schema": _INTERACTABLE_MASK_SCHEMA,
            },
        },
        # 토큰 여유
        # max_output_tokens=1536,
        # 시스템 규칙(선택): 반드시 JSON만
        # instructions="Return ONLY a JSON object that matches the provided schema.",
        timeout=timeout,
    )
    # 공식 라이브러리에서 권장하는 통합 프로퍼티
    # (README 예시: response.output_text)
    text = getattr(resp, "output_text", None) or ""
    if not text:
        # 디버깅 도움: 원시 output 확인
        print("Empty output_text. Raw response:", resp)
        raise RuntimeError("Empty model output_text")
    return json.loads(text)


@dataclass
class ProgramArgs:

    # Path to cache directory
    cachedir: str = ""

    # Path to map file
    mapfile: str = ""
    part_file: str = ""

    # Maximum number of detections to consider, per object
    max_detections_per_object: int = 10

    # Masking option
    masking_option: Literal["blackout", "red_outline", "red_mask", "none"] = "none"
    
    # LLaVA-related arguments
    llava_model_path: str = "liuhaotian/llava-v1.6-vicuna-7b"

    llava_mode: Literal["rigid", "part", "none"] = "none"

    part_reg: bool = False

    dataset_root: str = ""

    scene_name: str = ""


def get_classes_colors(classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in enumerate(classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_name] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors


def load_scene_map_results(loaded_data, scene_map):
    # Check the type of the loaded data to decide how to proceed
    if isinstance(loaded_data, dict) and "objects" in loaded_data:
        scene_map.load_serializable(loaded_data["objects"])
    elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with your expected type
        scene_map.load_serializable(loaded_data)
    else:
        raise ValueError("Unexpected data format in map file.")


def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """
    Crop the image with some padding

    Args:
        image: PIL image
        x1, y1, x2, y2: bounding box coordinates
        padding: padding around the bounding box

    Returns:
        image_crop: PIL image

    Implementation from the CFSLAM repo
    """
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """ Draw a red outline around the object i nan image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    red_outline = [255, 0, 0]

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red outlines around the object. The last argument "3" indicates the thickness of the outline.
    cv2.drawContours(image_np, contours, -1, red_outline, 3)

    # Optionally, add padding around the object by dilating the drawn contours
    kernel = np.ones((5, 5), np.uint8)
    image_np = cv2.dilate(image_np, kernel, iterations=1)
    
    image_pil = Image.fromarray(image_np)

    return image_pil


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
    
    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
        return None, None

    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None
    
    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop, mask_crop


def blackout_nonmasked_area(image_pil, mask):
    """ Blackout the non-masked area of an image"""
    # convert image to numpy array
    image_np = np.array(image_pil)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    black_image = Image.fromarray(black_image)
    return black_image


def red_masked_area(image_pil, mask):
    image_np = np.array(image_pil)
    # convert image to numpy array
    red_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    red_mask[mask] = [255, 0, 0]  # Green color where mask is True
    # convert back to pil image
    alpha = 0.2
    blended_image = cv2.addWeighted(image_np,1,red_mask,alpha,0)
    blended_image = Image.fromarray(blended_image)
    return blended_image


def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj, idx_obj_2=None):
    """ This is debug helper function that plots the images with the captions and masks overlaid and saves them to a directory. This way you can inspect exactly what the LLaVA model is captioning which image with the mask, and the mask confidence scores overlaid."""
    
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    if idx_obj_2 == None:
        plt.savefig(savedir / f"{idx_obj}.png")
    else:
        plt.savefig(savedir / f"{idx_obj}_{idx_obj_2}.png")
    plt.close()


def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)


def extract_node_captions(args, results):
    
    meta_file = os.path.join(args.dataset_root, 'metadata.csv')
    if os.path.exists(meta_file):
        with open(meta_file, encoding = 'utf-8') as f:
            meta_csv = np.loadtxt(f,str,delimiter = ",")
        for line in meta_csv:
            if args.scene_name.split('/')[0] in line and args.scene_name.split('/')[1] in line:
                camera_axis = line[2]
    else:
        camera_axis = 'Up'

    console = rich.console.Console()

    scene_map = MapObjectList()
    load_scene_map_results(results, scene_map)
    obj_cand = results['inter_id_candidate']
    print('Total Objects with rigid interaction: ', len(obj_cand))

    chat = LlavaModel16(
        model_path = args.llava_model_path,
        model_base = None, 
        conv_mode_input = None,
    )

    # Directories to save features and captions
    savedir_debug = Path(args.cachedir) / "cfslam_captions_llava_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)

    caption_dict_list = []
    for idx_obj, obj in tqdm(enumerate(scene_map)):
        if args.llava_mode == 'rigid':
            if idx_obj not in obj_cand:
                continue
        else:
            if idx_obj in obj_cand:
                continue
        mask_sum_num = np.array([(obj["xyxy"][idx][2] - obj["xyxy"][idx][0]) * (obj["xyxy"][idx][3] - obj["xyxy"][idx][1]) for idx in range(len(obj["mask"]))])
        # mask_sum_num = np.array([obj["mask"][idx].sum() for idx in range(len(obj["mask"]))]).astype(np.float64)
        conf = np.array(obj['conf'])
        if args.llava_mode == 'rigid':
            mask_sum_num *= conf
        else:
            mask_sum_num = conf
        sem_score = []
        if args.llava_mode == 'rigid':
            for idx in range(len(obj['class_name'])):
                obj_class = obj['class_name'][idx]
                if obj_class in ['kitchen cabinet', 'nightstand', 'cabinet', 'dresser', 'chest', 'drawer', 'closet', 'oven', 'door', 'trashcan', 'toilet', 'bathtub']:
                    sem_score.append(1.0)
                else:
                    sem_score.append(0.6)
        else:
            for idx in range(len(obj['class_name'])):
                sem_score.append(1.0)
        mask_sum_num *= np.array(sem_score)
        idx_most_conf = np.argsort(mask_sum_num)[::-1]

        captions = []
        low_confidences = []
        
        image_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []  # New list for masks
        
        num = 0
        for idx_det in tqdm(idx_most_conf):
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            if camera_axis == 'Left':
                image = image.rotate(-90, expand=True)
            xyxy = obj["xyxy"][idx_det]
            mask = obj["mask"][idx_det]
            class_name = obj['class_name'][idx_det]
            x1, y1, x2, y2 = xyxy
            if args.llava_mode == 'rigid':
                if x2 - x1 < 150 or y2 - y1 < 150:
                    args.masking_option = "red_outline"
                    padding = 100
                elif x2 - x1 < 200 or y2 - y1 < 200:
                    args.masking_option = "none"
                    padding = 100
                else:
                    args.masking_option = "none"
                    padding = 50
            else:
                if x2 - x1 < 150 or y2 - y1 < 150:
                    args.masking_option = "red_outline"
                    padding = 60
                else:
                    args.masking_option = "none"
                    padding = 30
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            if args.masking_option == "blackout":
                image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                image_crop_modified = draw_red_outline(image_crop, mask_crop)
            elif args.masking_option == "red_mask":
                image_crop_modified = red_masked_area(image_crop, mask_crop)
            else:
                image_crop_modified = image_crop  # No modification

            low_confidences.append(False)

            if args.llava_mode == 'rigid':
                if args.masking_option == "red_outline":
                    query = "Discribe the central object outlined by red."
                else:
                    query = "Describe the central household furniture in the image."
                # It might be a " + class_name + ", with confidence of " + f"{obj['conf'][idx_det]:.2f}, predicted by another model. You need to judge with the input image and the above reference information. If the prediction confidence is relatively low, you need to depend on the image.
            else:
                if args.masking_option == "red_outline":
                    query = "Discribe the object outlined by red. It might be a " + class_name + " predicted by others with confidence of" + f"{obj['conf'][idx_det]:.2f}. You should conbine this context and the input image to generate your answer"
                else:
                    query = "Describe the main object in the image."
                # It might be a " + class_name + ", with confidence of " + f"{obj['conf'][idx_det]:.2f}, predicted by another model. You need to judge with the input image and the above reference information. If the prediction confidence is relatively low, you need to depend on the image.
            console.print("[bold red]User:[/bold red] " + query)
            outputs = chat.infer(
                query = query,
                images = [image_crop_modified],
            )
            console.print("[bold green]LLaVA:[/bold green] " + outputs)
            captions.append(outputs)
            
            # For the LLava debug folder
            conf_value = obj['conf'][idx_det]
            image_list.append(image_crop_modified)
            caption_list.append(outputs)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)  # Add the cropped mask

            num += 1
            if num > args.max_detections_per_object:
                break

        caption_dict_list.append(
            {
                "id": idx_obj,
                "captions": captions,
                "low_confidences": low_confidences,
            }
        )
        
        # Again for the LLava debug folder
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, idx_obj)

    # Save the captions
    # Remove the "The central object in the image is " prefix from 
    # the captions as it doesnt convey and actual info
    if args.llava_mode == 'rigid':
        for item in caption_dict_list:
            item["captions"] = [caption.replace("The central object outlined by red", "") for caption in item["captions"]]
        for item in caption_dict_list:
            item["captions"] = [caption.replace("The central household furniture in the image", "") for caption in item["captions"]]
    else:
        for item in caption_dict_list:
            item["captions"] = [caption.replace("The central object outlined by red", "") for caption in item["captions"]]
        for item in caption_dict_list:
            item["captions"] = [caption.replace("The main object in the image", "") for caption in item["captions"]]
    # Save the captions to a json file
    with open(Path(args.cachedir) / "cfslam_llava_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)


def refine_node_captions(args):
    # Load the captions for each segment
    if not args.part_reg:
        caption_file = Path(args.cachedir) / "cfslam_llava_captions.json"
    else:
        caption_file = Path(args.cachedir) / "part_llava_captions.json"
    captions = None
    with open(caption_file, "r") as f:
        captions = json.load(f)
    
    # load the prompt
    gpt_messages = GPTPrompt().get_json()

    TIMEOUT = 80  # Timeout in seconds
    if not args.part_reg:
        responses_savedir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    else:
        responses_savedir = Path(args.cachedir) / "part_gpt-4_responses"
    responses_savedir.mkdir(exist_ok=True, parents=True)

    responses = []
    unsucessful_responses = 0

    # loop over every object
    for _i in trange(len(captions)):
        if len(captions[_i]) == 0:
            continue
        
        # Prepare the object prompt 
        _dict = {}
        _caption = captions[_i]
        _dict["id"] = _caption["id"]
        _dict["captions"] = _caption["captions"]
        
        # Make and format the full prompt
        preds = json.dumps(_dict, indent=0)

        start_time = time.time()
    
        curr_chat_messages = gpt_messages[:]
        curr_chat_messages.append({"role": "user", "content": preds})
        # chat_completion = client.completions.create(
        #     # model="gpt-3.5-turbo",
        #     model="gpt-4",
        #     prompt=curr_chat_messages,
        #     timeout=TIMEOUT,  # Timeout in seconds
        # )
        resp_text = _ask_gpt4_text(curr_chat_messages, TIMEOUT)


        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT:
            print("Timed out exceeded!")
            _dict["response"] = "FAIL"
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
            responses.append(json.dumps(_dict))
            unsucessful_responses += 1
            exit(1)
        
        # # count unsucessful responses
        # if "invalid" in chat_completion["choices"][0]["message"]["content"].strip("\n"):
        #     unsucessful_responses += 1
            
        # # print output
        # prjson([{"role": "user", "content": preds}])
        # print(chat_completion["choices"][0]["message"]["content"])
        # print(f"Unsucessful responses so far: {unsucessful_responses}")
        # _dict["response"] = chat_completion["choices"][0]["message"]["content"].strip("\n")
        if "invalid" in (resp_text or "").strip("\n"):
            unsucessful_responses += 1
        prjson([{"role": "user", "content": preds}])
        print(resp_text)
        print(f"Unsucessful responses so far: {unsucessful_responses}")
        _dict["response"] = (resp_text or "").strip("\n")
        

        # save the response
        responses.append(json.dumps(_dict))
        save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")

    if not args.part_reg:
        with open(Path(args.cachedir) / "cfslam_gpt-4_responses.pkl", "wb") as f:
            pkl.dump(responses, f)
    else:
        with open(Path(args.cachedir) / "part_gpt-4_responses.pkl", "wb") as f:
            pkl.dump(responses, f)


def filter_rigid_objects(args, results, part_results):
    obj_cand = results['inter_id_candidate']
    scene_map = MapObjectList()
    load_scene_map_results(results, scene_map)
    part_map = MapObjectList()
    load_scene_map_results(part_results, part_map)

    response_dir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses = []
    object_tags = []
    also_indices_to_remove = [] # indices to remove if the json file does not exist
    for idx in obj_cand:
        # check if the json file exists first 
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                _d["response"] = json.loads(_d["response"])
            except json.JSONDecodeError:
                _d["response"] = {
                    'summary': f'GPT4 json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])
    for i, tag in enumerate(object_tags):
        if ' with ' in tag:
            object_tags[i] = tag.split(' with ')[0]
        elif ' on ' in tag:
            object_tags[i] = tag.split(' on ')[-1]
        elif ' and ' in tag:
            object_tags[i] = tag.split(' and ')[0]
    
    if not (Path(args.cachedir) / "rigid_interactable_object_mask.json").exists():
        TIMEOUT = 25
        DEFAULT_PROMPT = """
            Below is a list containing some predicted specific object tags of furnitures in the same room. 
            You should select which furnitures belong to categories of something having space for storage: drawer, cabinet, kitchen countertop, dresser, wardrobe, nightstand, TV stand, trashcan;
            something for open or close: door, window;
            some appliances for operating: fridge, oven, coffee maker;
            something for controlling the water flow: sink,  toilet, bathtub, radiator.
            (having similar meanings with these pre-defined categories, having knobs, handles, buttons to interact with).
            Please produce a JSON string (and nothing else), with keys 'filter_mask' and 'reason'.
            In the key 'filter_mask', store a list with the same length of the input object tags list.
            Each item of the list is True, if you think this furniture belongs to the categories described above, otherwise False.
            If you meet invalid object tag item, directly set the corresponding output False.
            If you meet some vague tags with too broad definition (e.g. bedroom furniture / bathroom / container / box), set False.
            If the tags already contain part labels (e.g. button, handle, knob), set False.
            If you meet two conflict tags linked by 'and': if BOTH of them may be with knobs, buttons or handles to interact with, set True, otherwise False.
            In the key 'reason', provide a list with corresponding reasons for your choices.
        """

        start_time = time.time()
        # chat_completion = client.completions.create(
        #     # model="gpt-3.5-turbo",
        #     model="gpt-4",
        #     prompt=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + json.dumps(object_tags)}],
        #     timeout=60,  # Timeout in seconds
        # )
        resp_text = _ask_gpt4_text(
            [{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + json.dumps(object_tags)}],
            60,
        )


        elapsed_time = time.time() - start_time
        output_dict = {}
        output_dict['object_tags'] = object_tags
        if elapsed_time > TIMEOUT:
            print("Timed out exceeded!")
            output_dict["filter_mask"] = "FAIL"
            output_dict["reason"] = "FAIL"
            exit(1)
        else:
            # try:
            #     # Attempt to parse the output as a JSON
            #     chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
            #     # If the output is a valid JSON, then add it to the output dictionary
            #     output_dict["filter_mask"] = chat_output_json["filter_mask"]
            #     output_dict["reason"] = chat_output_json["reason"]
            # except:
            #     output_dict["filter_mask"] = "FAIL"
            #     output_dict["reason"] = "FAIL"
            try:
                chat_output_json = json.loads(resp_text)
                output_dict["filter_mask"] = chat_output_json["filter_mask"]
                output_dict["reason"] = chat_output_json["reason"]
            except:
                output_dict["filter_mask"] = "FAIL"
                output_dict["reason"] = "FAIL"
        

        # Saving the output
        print("Saving object masks to file...")
        with open(Path(args.cachedir) / "rigid_interactable_object_mask.json", "w") as f:
            json.dump(output_dict, f, indent=4)
    else:
        output_dict = json.load(open(Path(args.cachedir) / "rigid_interactable_object_mask.json", "r"))
    
    filter_mask = output_dict["filter_mask"]
    rigid_inter_id_candidate = []
    for i in range(len(filter_mask)):
        if filter_mask[i]:
            rigid_inter_id_candidate.append(obj_cand[i])
            scene_map[obj_cand[i]]['refined_obj_tag'] = object_tags[i]
    # parts
    part_inter_id_candidate = []
    for idx in obj_cand:
        obj = scene_map[idx]
        if idx not in rigid_inter_id_candidate:
            obj['connected_parts'] = []
        else:
            for part_idx in obj['connected_parts']:
                part_inter_id_candidate.append(part_idx)
        
    updated_results = {
        'objects': scene_map.to_serializable(),
        'cfg': results['cfg'],
        'class_names': results['class_names'],
        'class_colors': results['class_colors'],
        'inter_id_candidate': list(set(rigid_inter_id_candidate))
    }    

    updated_part_results = {
        'objects': part_map.to_serializable(),
        'cfg': part_results['cfg'],
        'class_names': part_results['class_names'],
        'class_colors': part_results['class_colors'],
        'part_inter_id_candidate': list(set(part_inter_id_candidate))
    }    

    return updated_results, updated_part_results


def extract_part_captions(args, results, part_results):
    
    meta_file = os.path.join(args.dataset_root, 'metadata.csv')
    if os.path.exists(meta_file):
        with open(meta_file, encoding = 'utf-8') as f:
            meta_csv = np.loadtxt(f,str,delimiter = ",")
        for line in meta_csv:
            if args.scene_name.split('/')[0] in line and args.scene_name.split('/')[1] in line:
                camera_axis = line[2]
    else:
        camera_axis = 'Up'

    obj_cand = results['inter_id_candidate']
    scene_map = MapObjectList()
    load_scene_map_results(results, scene_map)
    part_map = MapObjectList()
    load_scene_map_results(part_results, part_map)

    console = rich.console.Console()

    chat = LlavaModel16(
        model_path = args.llava_model_path,
        model_base = None, 
        conv_mode_input = None,
    )

    # Directories to save features and captions
    savedir_debug = Path(args.cachedir) / "part_captions_llava_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)

    caption_dict_list = []
    total_part_idx = []
    for idx_obj, obj in tqdm(enumerate(scene_map)):
        if idx_obj not in obj_cand:
            continue
        obj_name = obj['refined_obj_tag']
        for part_idx in obj['connected_parts']:
            if part_idx in total_part_idx:
                continue
            part = part_map[part_idx]
            total_part_idx.append(part_idx)
            mask_sum_num = 1. / np.array([part["mask"][idx].sum() for idx in range(len(part["mask"]))])
            conf = np.array(part['conf'])
            mask_sum_num *= conf
            idx_most_conf = np.argsort(mask_sum_num)[::-1]

            captions = []
            low_confidences = []
            
            image_list = []
            caption_list = []
            confidences_list = []
            low_confidences_list = []
            mask_list = []  # New list for masks
            
            num = 0
            for idx_det in tqdm(idx_most_conf):
                image = Image.open(part["color_path"][idx_det]).convert("RGB")
                if camera_axis == 'Left':
                    image = image.rotate(-90, expand=True)
                xyxy = part["xyxy"][idx_det]
                mask = part["mask"][idx_det]
                class_name = part['class_name'][idx_det]

                padding = 40
                x1, y1, x2, y2 = xyxy
                image_crop, mask_crop0 = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
                if args.masking_option == "blackout":
                    image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop0)
                elif args.masking_option == "red_outline":
                    image_crop_modified = draw_red_outline(image_crop, mask_crop0)
                else:
                    image_crop_modified = image_crop  # No modification

                padding = 1.5 * min((x2-x1), (y2-y1))
                image_crop1, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
                if args.masking_option == "blackout":
                    image_crop_modified1 = blackout_nonmasked_area(image_crop1, mask_crop)
                elif args.masking_option == "red_outline":
                    image_crop_modified1 = draw_red_outline(image_crop1, mask_crop)
                else:
                    image_crop_modified1 = image_crop1  # No modification
                
                padding = 2.5 * min((x2-x1), (y2-y1))
                image_crop2, mask_crop2 = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
                if args.masking_option == "blackout":
                    image_crop_modified2 = blackout_nonmasked_area(image_crop2, mask_crop2)
                elif args.masking_option == "red_outline":
                    image_crop_modified2 = draw_red_outline(image_crop2, mask_crop2)
                else:
                    image_crop_modified2 = image_crop2  # No modification

                concat_image = concatenate_images_horizontal([image_crop_modified, image_crop_modified1, image_crop_modified2], 20)

                low_confidences.append(False)

                query = '''Describe the item outlined by red.
                You are given three images showing a same item with three scales to give you helpful background context.
                In the left, the item is zoomed-in the most, while in the right, the item is zoomed-in the least.
                That is, from the left to the right, the item itself becomes smaller, and its background information becomes larger.
                You should combine this three scales to make the recognition.
                Because all three images capture the same item, you just need to generate a single summarized recognition result for the item, do not separately generate it for three images. 
                To make some hint to you, your answer can be selected from the four categories: 'knob', 'button', 'handle', 'others'.
                To make the choice, first, decide whether the item is a part really belonging a ''' + obj_name + \
                ''' via the multi-scaled images.
                If not, you can answer 'others' and give your judgement.
                If you think this item is a part of the object, choose it from the first three categories and make detailed descriptions. 
                If it has a round shape, looks like to pull or rotate, and lies on a wooden furniture or oven, choose 'knob'.
                If it has a round shape, looks like to press, and lies in bathtub or sink or on other electric appliance, choose 'button'.
                If it has a cylinder shape, looks like to pull or rotate, and lies on wooden furniture, door, window, choose 'handle'.
                Format your answer with the start: 'The item outlined by red is [YOUR CHOICE]. Description: ...' '''
  
                console.print("[bold red]User:[/bold red] " + query)
                outputs = chat.infer(
                    query = query,
                    images = [concat_image],
                )
                console.print("[bold green]LLaVA:[/bold green] " + outputs)
                captions.append(outputs)
                
                # For the LLava debug folder
                conf_value = part['conf'][idx_det]
                image_list.append(image_crop_modified1)
                caption_list.append(outputs)
                confidences_list.append(conf_value)
                low_confidences_list.append(low_confidences[-1])
                mask_list.append(mask_crop)  # Add the cropped mask

                num += 1
                if num > args.max_detections_per_object:
                    break

            caption_dict_list.append(
                {
                    "id": part_idx,
                    "captions": captions,
                    "low_confidences": low_confidences,
                }
            )
            
            # Again for the LLava debug folder
            if len(image_list) > 0:
                plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, part_idx)

    # Save the captions
    # Remove the "The central object in the image is " prefix from 
    # the captions as it doesnt convey and actual info
    for item in caption_dict_list:
        item["captions"] = [caption.replace("The item outlined by red is", "") for caption in item["captions"]]
    # Save the captions to a json file
    with open(Path(args.cachedir) / "part_llava_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)


def build_rigid_funcgraph(args, results, part_results):

    obj_cand = results['inter_id_candidate']
    part_cand = part_results['part_inter_id_candidate']
    scene_map = MapObjectList()
    load_scene_map_results(results, scene_map)
    part_map = MapObjectList()
    load_scene_map_results(part_results, part_map)

    response_dir = Path(args.cachedir) / "part_gpt-4_responses"
    responses = []
    part_summarys = {}
    part_tags = {}
    also_indices_to_remove = [] # indices to remove if the json file does not exist
    for idx in range(len(part_map)):
        # check if the json file exists first 
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                _d["response"] = json.loads(_d["response"])
            except json.JSONDecodeError:
                _d["response"] = {
                    'summary': f'GPT4 json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            part_summarys[_d["id"]] = _d["response"]["summary"]
            part_tags[_d["id"]] = _d["response"]["object_tag"]

    TIMEOUT = 80  # timeout in seconds
    relations = []
    if not (Path(args.cachedir) / "cfslam_object_rigid_relations.json").exists():
        # pruning using GPT-4 and ensuring the functional edges
        for i in obj_cand:
            obj = scene_map[i]
            for part_idx in list(set(obj['connected_parts'])):
                input_dict = {
                    "object": {
                        "id": i,
                        "tag": obj['refined_obj_tag'],
                    },
                    "part":  {
                        "id": part_idx,
                        "tag": part_tags[part_idx],
                    }
                }
           
                print(f"{input_dict['object']['tag']}, {input_dict['part']['tag']}")

                input_json_str = json.dumps(input_dict)

                # Default prompt
                DEFAULT_PROMPT = """
                The input is a JSON describing an household object "object" and a part of the object for human interaction "part".
                As an agent specialized in analyzing the relationship between different household objects and their interactable parts, 
                you need to produce a JSON string (and nothing else), with three keys: "connection_tag", "function", and "reason".

                Each "object" and "part" JSON has the following field:
                tag: a brief description of the object or the part;

                You need to produce "connection_tag", "function", and "reason" for filtering invalid parts and inferring the function of the part to the object if this part is interacted.

                First, please produce the "connection_tag".
                If you think the part seems belonging to the object as an interactable part, determine "connection_tag" as True.
                Otherwise it should be False.
                The part needs functional connection with the object, and should manipulate the object.
                For example, a knob or handle is reasonable as a functional part of a cabinet, button is not.
                If the part tag contains "invalid", set False.
                If the part tag is too vague or broad, e.g. directly with the word "part", set the "connection_tag" False.

                Second, you need to determine the "function" of the part if the "connection_tag" is True.
                "function" describes the functional connection between this "part" and the "object".
                For example, a knob at the side of a chest is used for pulling to open the chest;
                a knob at the side of an oven is used for rotating to adjust its settings;
                a knob or handle of a trashcan is used for opening it;
                a handle at the side of a refrigerator is used for pulling to open it;
                a button at the top of a toilet is used for pushing for flushing;
                a knob or a button on a sink, or bathtub is used for rotating or pressing to control the water;
                a knob or a handle on a kitchen countertop or bathroom vanity is used for pulling a drawer of it.
                
                Finally, produce the "reason" list that explains why you produce such "connection_tag" and "function".
                """
                # Note that if you meet with object tags like television stand and entertainment center, knob and handles are likely for pulling drawers on them.
                start_time = time.time()
                # chat_completion = client.completions.create(
                #     # model="gpt-3.5-turbo",
                #     model="gpt-4",
                #     prompt=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                #     timeout=TIMEOUT,  # Timeout in seconds
                # )
                resp_text = _ask_gpt4_text(
                    [{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                    TIMEOUT,
                )


                elapsed_time = time.time() - start_time
                output_dict = input_dict
                if elapsed_time > TIMEOUT:
                    print("Timed out exceeded!")
                    output_dict["connection_tag"] = "FAIL"
                    output_dict["function"] = 'FAIL'
                    output_dict["reason"] = 'FAIL'
                    exit(1)
                else:
                    # try:
                    #     # Attempt to parse the output as a JSON
                    #     chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
                    #     # If the output is a valid JSON, then add it to the output dictionary
                    #     output_dict["connection_tag"] = chat_output_json["connection_tag"]
                    #     output_dict["function"] = chat_output_json["function"]
                    #     output_dict["reason"] = chat_output_json["reason"]
                    # except:
                    #     output_dict["connection_tag"] = "FAIL"
                    #     output_dict["function"] = 'FAIL'
                    #     output_dict["reason"] = 'FAIL'
                    #     exit(1)
                    try:
                        chat_output_json = json.loads(resp_text)
                        output_dict["connection_tag"] = chat_output_json["connection_tag"]
                        output_dict["function"] = chat_output_json["function"]
                        output_dict["reason"] = chat_output_json["reason"]
                    except:
                        output_dict["connection_tag"] = "FAIL"
                        output_dict["function"] = "FAIL"
                        output_dict["reason"] = "FAIL"
                        exit(1)


                relations.append(output_dict)

        # Saving the output
        print("Saving object relations to file...")
        with open(Path(args.cachedir) / "cfslam_object_rigid_relations.json", "w") as f:
            json.dump(relations, f, indent=4)
    else:
        relations = json.load(open(Path(args.cachedir) / "cfslam_object_rigid_relations.json", "r"))

    scenegraph_edges = []
  
    for i in obj_cand:
        scene_map[i]['connected_parts'] = []
    new_part_cand = []
    # pruning invalid parts and determine the edges
    for rel in relations:
        if rel["connection_tag"]:
            part_idx = rel['part']['id']
            obj_idx = rel['object']['id']
            new_part_cand.append(part_idx)
            part_map[part_idx]['refined_obj_tag'] = part_tags[part_idx]
            scene_map[obj_idx]['connected_parts'].append(part_idx)
            scenegraph_edges.append((obj_idx, part_idx, -1, rel["function"]))
    
    for i in obj_cand:
        scene_map[i]['connected_parts'] = list(set(scene_map[i]['connected_parts']))

    updated_part_results = {
        'objects': part_map.to_serializable(),
        'cfg': part_results['cfg'],
        'class_names': part_results['class_names'],
        'class_colors': part_results['class_colors'],
        'part_inter_id_candidate': list(set(new_part_cand))
    }   

    save_path = args.part_file
    
    with gzip.open(save_path, "wb") as f:
        pkl.dump(updated_part_results, f)
    print(f"Saved full point cloud to {save_path}") 

    updated_results = {
        'objects': scene_map.to_serializable(),
        'cfg': results['cfg'],
        'class_names': results['class_names'],
        'class_colors': results['class_colors'],
        'inter_id_candidate': list(set(obj_cand))
    }    

    save_path = args.mapfile
    
    with gzip.open(save_path, "wb") as f:
        pkl.dump(updated_results, f)
    print(f"Saved full point cloud to {save_path}")

    scenegraph_edges = list(set(scenegraph_edges))

    with open(Path(args.cachedir).parent / "cfslam_funcgraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)
    
    return updated_results, updated_part_results, scenegraph_edges


def filter_remote_interactable_objects(args, results):
    scene_map = MapObjectList()
    load_scene_map_results(results, scene_map)
    obj_cand = results['inter_id_candidate']

    response_dir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses = []
    object_tags = []
    id_list = []
    for idx in range(len(scene_map)):
        if idx in obj_cand:
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                _d["response"] = json.loads(_d["response"])
            except json.JSONDecodeError:
                _d["response"] = {
                    'summary': f'GPT4 json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])
            id_list.append(_d["id"])
    # Remove segments that correspond to "invalid" tags
    indices_to_remove = [i for i in range(len(responses)) if object_tags[i].lower() in ["fail", "invalid"]]
    # combine with also_indices_to_remove and sort the list
    indices_to_remove = list(set(indices_to_remove))
    # List of tags in original scene map that are in the pruned scene map
    segment_ids_to_retain = [i for i in range(len(responses)) if i not in indices_to_remove]
    print(f"Removed {len(indices_to_remove)} segments")
    # Filtering responses based on segment_ids_to_retain
    responses = [responses[i] for i in range(len(responses)) if i in segment_ids_to_retain]
    # Assuming each response dictionary contains an 'object_tag' key for the object tag.
    # Extract filtered object tags based on filtered_responses
    object_tags = [resp['response']['object_tag'] for resp in responses]
    pruned_id_list = [resp['id'] for resp in responses]

    for i, tag in enumerate(object_tags):
        if ' with ' in tag:
            object_tags[i] = tag.split(' with ')[0]
        elif ' on ' in tag:
            object_tags[i] = tag.split(' on ')[-1]
        elif ' and ' in tag:
            object_tags[i] = tag.split(' and ')[0]
    
    if not (Path(args.cachedir) / "interactable_object_mask.json").exists():

        DEFAULT_PROMPT = """
            Below is a list containing some predicted object tags of household objects in the same room. 
            As a specialist agent analysing functional connections among different objects, you need to determine which objects could be operated remotely (e.g. TV, air conditioner, oven, refrigerator, ceiling light) or could operate other items remotely (e.g. remote control, switch, electric outlet).
            Typically they are related with electricity, with functional connection of remote controling or power supplying.
            Note that part tags like button, handle, knob, bar, or balustrade, they are not objects of interest.
            Some too vague tags are not objects of interest, e.g. bin, container, box, device.
            Please produce a JSON string (and nothing else), with keys 'mask' and 'reason'.
            In the key 'mask', store a list with the EXACTLY SAME length of the input list, claiming True or False whether each object is interactable.
            In the key 'reason', illustrate the cooresponding reasons of each choice.
        """

        # chat_completion = client.completions.create(
        #     # model="gpt-3.5-turbo",
        #     model="gpt-4",
        #     prompt=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + json.dumps(object_tags)}],
        #     timeout=60,  # Timeout in seconds
        # )
        resp_json = _ask_json_response(
            [{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + json.dumps(object_tags)}],
            timeout=60,
            model="gpt-4o-mini",
        )

        output_dict = {'id': pruned_id_list, 'object_tags': object_tags}
        output_dict["mask"] = resp_json["mask"]
        output_dict["reason"] = resp_json["reason"]
        # try:
        #     # Attempt to parse the output as a JSON
        #     chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
        #     # If the output is a valid JSON, then add it to the output dictionary
        #     output_dict["mask"] = chat_output_json["mask"]
        #     output_dict["reason"] = chat_output_json["reason"]
        # except:
        #     output_dict["mask"] = "FAIL"
        #     output_dict["reason"] = "FAIL"
        #     print('Failed!')
        #     exit(1)
        try:
            chat_output_json = json.loads(resp_text)
            output_dict["mask"] = chat_output_json["mask"]
            output_dict["reason"] = chat_output_json["reason"]
        except:
            output_dict["mask"] = "FAIL"
            output_dict["reason"] = "FAIL"
            print("Failed!")
            exit(1)
        


        # Saving the output
        print("Saving object masks to file...")
        with open(Path(args.cachedir) / "interactable_object_mask.json", "w") as f:
            json.dump(output_dict, f, indent=4)
    else:
        output_dict = json.load(open(Path(args.cachedir) / "interactable_object_mask.json", "r"))
    
    for i in range(len(output_dict["mask"])):
        if output_dict["mask"][i]:
            obj_id = output_dict['id'][i]
            scene_map[obj_id]['refined_obj_tag'] = output_dict['object_tags'][i]
            obj_cand.append(obj_id)

    updated_results = {
        'objects': scene_map.to_serializable(),
        'cfg': results['cfg'],
        'class_names': results['class_names'],
        'class_colors': results['class_colors'],
        'inter_id_candidate': list(set(obj_cand))
    }    

    save_path = Path(args.mapfile)
    
    with gzip.open(save_path, "wb") as f:
        pkl.dump(updated_results, f)
    print(f"Saved full point cloud to {save_path}")

    return updated_results


def build_object_funcgraph(args, results, func_edges):
    obj_cand = results['inter_id_candidate']
    scene_map = MapObjectList()
    load_scene_map_results(results, scene_map)

    obj_tags = []
    for i, idx1 in enumerate(obj_cand):
        obj_tags.append(scene_map[idx1]['refined_obj_tag'])

    categories = []
    if not (Path(args.cachedir) / "functional_object_categories.json").exists():
        
        input_dict = {"objects": obj_tags,}

        input_json_str = json.dumps(input_dict)

        # Default prompt
        DEFAULT_PROMPT = """
        The input is a list of JSONs describing several "objects".
        You need to catagorize them with (1) interactable elements; (2) other objects.
        To be more specific, (1) interactable elements include switch, remote control, electric outlet etc., which decide other objects' states alteration and power supply.
        (2) other objects include TV, ceiling light, electric applicance etc. whose states are decided by interactable elements.
        You need to produce a JSON string (and nothing else), with two keys: "categories", and "reason".
        "categories" is a list with the same length as "objects".
        "reason" is a list of the reason why you choose the categories.
        """

        # chat_completion = client.completions.create(
        #     # model="gpt-3.5-turbo",
        #     model="gpt-4",
        #     prompt=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
        #     timeout=60,  # Timeout in seconds
        # )
        resp_text = _ask_gpt4_text(
            [{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
            60,
        )


        output_dict = input_dict
        # try:
        #     # Attempt to parse the output as a JSON
        #     chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
        #     # If the output is a valid JSON, then add it to the output dictionary
        #     output_dict["categories"] = chat_output_json["categories"]
        #     output_dict["reason"] = chat_output_json["reason"]
        # except:
        #     output_dict["categories"] = "FAIL"
        #     output_dict["reason"] = "FAIL"
        #     exit(1)
        try:
            chat_output_json = json.loads(resp_text)
            output_dict["categories"] = chat_output_json["categories"]
            output_dict["reason"] = chat_output_json["reason"]
        except:
            output_dict["categories"] = "FAIL"
            output_dict["reason"] = "FAIL"
            exit(1)


        categories = output_dict["categories"]
        # Saving the output
        print("Saving object categories to file...")
        with open(Path(args.cachedir) / "functional_object_categories.json", "w") as f:
            json.dump(output_dict, f, indent=4)
    else:
        categories = json.load(open(Path(args.cachedir) / "functional_object_categories.json", "r"))["categories"]

    relations = []
    if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
        for i, idx1 in enumerate(obj_cand):
            # only for interactable elements
            if 'other' in categories[i]:
                continue
            obj1 = scene_map[idx1]
            _bbox1 = obj1["bbox"]
            input_dict = {
                "interactable element": {
                    "id": idx1,
                    "tag": obj1['refined_obj_tag'],
                },
                "objects": []
            }
            for j, idx2 in enumerate(obj_cand):
                # only for objects
                if 'interactable' in categories[j]:
                    continue
                obj2 = scene_map[idx2]
                _bbox2 = obj2["bbox"]
                input_dict["objects"].append({
                    "id": idx2,
                    "tag": obj2['refined_obj_tag'],
                })
            print(input_dict["interactable element"]["tag"])
            if len(input_dict["objects"]) == 0:
                continue
            input_json_str = json.dumps(input_dict)
            # Default prompt
            DEFAULT_PROMPT = """
            The input is a list of JSONs describing a interactable element "interactable element" and a list of "objects" with their "id", "bbox_center", and "tag".
            You need to produce a JSON string (and nothing else), with two keys: "functional_connection", and "reason".

            As an agent specialized in analyzing interaction between different household objects,
            please produce an "functional_connection" field that best describes the functional connection between the "interactable element" and all the "objects".
            It should be a list with the same length of "objects".
            Such functional connection is defined as the "interactable element" commonly determines the state change of the object.
            For example, a remote controls a television; 
            a switch controls a ceiling light or lamp;
            an electric outlet provides power for electric appliances including lights, lamps, ovens, TVs, etc.
            Note that typically, a remote cannot control appliances like lamp, light fixture;
            If the "interactable element" don't have the funtional connection with the object, the item corresponding to this object in the "functional_connection" field should be "NULL".
            If the "interactable element" is a mixed caption like 'A/B', "functional_connection" should be "relation A / relation B" for each caption separately.

            After producing the "functional_connection" field, produce a "reason" field that explains why the produced "functional_connection" field is the best.
            It also should be a list with the same length of "objects".
            """

            # chat_completion = client.completions.create(
            #     # model="gpt-3.5-turbo",
            #     model="gpt-4",
            #     prompt=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
            #     timeout=60,  # Timeout in seconds
            # )
            resp_text = _ask_gpt4_text(
                [{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                60,
            )


            output_dict = input_dict
            # try:
            #     # Attempt to parse the output as a JSON
            #     chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
            #     # If the output is a valid JSON, then add it to the output dictionary
            #     output_dict["object_relation"] = chat_output_json["object_relation"]
            #     output_dict["reason"] = chat_output_json["reason"]
            # except:
            #     output_dict["object_relation"] = "FAIL"
            #     output_dict["reason"] = "FAIL"
            #     exit(1)
            try:
                chat_output_json = json.loads(resp_text)
                output_dict["object_relation"] = chat_output_json["object_relation"]
                output_dict["reason"] = chat_output_json["reason"]
            except:
                output_dict["object_relation"] = "FAIL"
                output_dict["reason"] = "FAIL"
                exit(1)

                
            relations.append(output_dict)

        # Saving the output
        print("Saving object relations to file...")
        with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
            json.dump(relations, f, indent=4)
    else:
        relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))

    edges_to_remove = []
    for idx, edge in enumerate(func_edges):
        if edge[1] == -1:
            edges_to_remove.append(idx)
    if len(edges_to_remove) > 0:
        for index in sorted(edges_to_remove, reverse=True):
            del func_edges[index]

    if len(relations) > 0:
        for rel in relations:
            for i in range(len(rel['objects'])):
                if "NULL" not in rel["object_relation"][i]:
                    func_edges.append((rel["interactable element"]["id"], -1, rel["objects"][i]["id"], rel["object_relation"][i]))

        func_edges = list(set(func_edges))
        print(f"Created 3D funcgraph with {len(func_edges)} edges")
        with open(Path(Path(args.cachedir).parent / "cfslam_funcgraph_edges.pkl"), "wb") as f:
            pkl.dump(func_edges, f)

    return func_edges


def concatenate_images_horizontal(images, dist_images):
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in images) + dist_images * (len(images) - 1)
    # calc max height from imgs
    height = max(img.height for img in images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + dist_images

    return new_img


def prune_graph(args, results, func_edges):
    from openfungraph.llava.llava_model_16 import LlavaModel16
    from openfungraph.slam.slam_classes import MapObjectList

    meta_file = os.path.join(args.dataset_root, 'metadata.csv')
    if os.path.exists(meta_file):
        with open(meta_file, encoding = 'utf-8') as f:
            meta_csv = np.loadtxt(f,str,delimiter = ",")
        for line in meta_csv:
            if args.scene_name.split('/')[0] in line and args.scene_name.split('/')[1] in line:
                camera_axis = line[2]
    else:
        camera_axis = 'Up'

    scene_map = MapObjectList()
    load_scene_map_results(results, scene_map)
    
    # 0: inter_ele, 1: -1 (part), 2: obj, 3: label, +4:confidence
    for idx, edge in enumerate(func_edges):
        func_edges[idx] = list(edge)

    console = rich.console.Console()

    chat = LlavaModel16(
        model_path = args.llava_model_path,
        model_base = None, 
        conv_mode_input = None,
    )

    # Directories to save features and captions
    savedir_debug = Path(args.cachedir) / "graph_prune"
    savedir_debug.mkdir(exist_ok=True, parents=True)

    phase_2_dict = {}
    phase_3_dict = {}
    for edge in func_edges:
        if edge[1] == -1:
            # remote cases, setting confidence score
            inter_ele_idx = edge[0]
            obj_idx = edge[2]
            inter_ele = scene_map[inter_ele_idx]
            obj = scene_map[obj_idx]
            if 'outlet' in inter_ele['refined_obj_tag'] or 'power' in inter_ele['refined_obj_tag'] or 'switch' in inter_ele['refined_obj_tag']:
                # phase 1 by distance
                inter_ele_center = inter_ele['bbox'].center
                obj_center = obj['bbox'].center
                edge.append(1. / np.linalg.norm(inter_ele_center - obj_center))
            if 'outlet' in inter_ele['refined_obj_tag'] or 'power' in inter_ele['refined_obj_tag'] or 'switch' in inter_ele['refined_obj_tag']:
                # phase 2 inferred by LLAVA
                # inter_ele centered
                if inter_ele_idx not in phase_2_dict.keys():
                    phase_2_dict[inter_ele_idx] = [obj_idx]
                else:
                    if obj_idx not in phase_2_dict[inter_ele_idx]:
                        phase_2_dict[inter_ele_idx].append(obj_idx)
            if 'remote' in inter_ele['refined_obj_tag']:
                # phase 3 inferred by LLAVA
                # obj centered
                if obj_idx not in phase_3_dict.keys():
                    phase_3_dict[obj_idx] = [inter_ele_idx]
                else:
                    if inter_ele_idx not in phase_3_dict[obj_idx]:
                        phase_3_dict[obj_idx].append(inter_ele_idx)
    
    for inter_ele_idx, obj_idxs in phase_2_dict.items():
        inter_ele = scene_map[inter_ele_idx]
        conf = np.array(inter_ele['conf'])
        sem_score = []
        for idx in range(len(inter_ele['class_name'])):
            inter_ele_class = inter_ele['class_name'][idx]
            if 'switch' in inter_ele_class or 'panel' in inter_ele_class:
                sem_score.append(1.0)
            else:
                sem_score.append(0.6)
        idx_most_conf = np.argsort(conf*np.array(sem_score))[::-1]
        idx_det = idx_most_conf[0]
        image = Image.open(inter_ele["color_path"][idx_det]).convert("RGB")
        if camera_axis == 'Left':
            image = image.rotate(-90, expand=True)
        xyxy = inter_ele["xyxy"][idx_det]
        mask = inter_ele["mask"][idx_det]
        x1, y1, x2, y2 = xyxy
        padding = 30
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
        image_crop_modified = image_crop  # No modification

        captions_list = []
        obj_centers_list = []
        obj_names_list = []
        for obj_idx in obj_idxs:
            obj = scene_map[obj_idx]
            conf = np.array(obj['conf'])
            obj_centers_list.append(list(obj['bbox'].center))
            obj_names_list.append(obj['refined_obj_tag'])
            sem_score = []
            for idx in range(len(obj['class_name'])):
                obj_class = obj['class_name'][idx]
                if obj_class in obj['refined_obj_tag']:
                    sem_score.append(1.0)
                else:
                    sem_score.append(0.6)
            idx_most_conf = np.argsort(conf*np.array(sem_score))[::-1]
            idx_det = idx_most_conf[0]
            obj_image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            if camera_axis == 'Left':
                obj_image = obj_image.rotate(-90, expand=True)
            xyxy = obj["xyxy"][idx_det]
            mask = obj["mask"][idx_det]
            x1, y1, x2, y2 = xyxy
            padding = 30
            obj_image_crop, mask_crop = crop_image_and_mask(obj_image, mask, x1, y1, x2, y2, padding=padding)
            obj_image_crop_modified = obj_image_crop
            concat_image = concatenate_images_horizontal([image_crop_modified, obj_image_crop_modified], 20)
            query = '''
                You are given two images. The left image depicts a
            ''' + inter_ele['refined_obj_tag'] +\
            '''
                , while the right shows a 
            ''' + obj['refined_obj_tag'] +\
            '''
                . Please illustrate if the 
            ''' + inter_ele['refined_obj_tag'] +\
            '''
                 in the left image controls the  
            ''' + obj['refined_obj_tag'] +\
            '''
                 in the right image. You should provide your illustration, reason, and the confidence score between 0 to 1.
                 Format your answer with: Illustration: ...  Confidence: ... Reason: ...
            '''
            console.print("[bold red]User:[/bold red] " + query)
            outputs = chat.infer(
                query = query,
                images = [concat_image],
            )
            console.print("[bold green]LLaVA:[/bold green] " + outputs)
            captions_list.append(outputs)
        
        if not (Path(savedir_debug) / ("prune_graph_" + str(inter_ele_idx) +".json")).exists():
            input_dict = {
                'inter_ele_idx': inter_ele_idx,
                'inter_ele_center': list(inter_ele['bbox'].center),
                'inter_ele_name': inter_ele['refined_obj_tag'],
                'obj_idxs': obj_idxs,
                'obj_centers': obj_centers_list,
                'obj_names': obj_names_list,
                'captions': captions_list
            }
            input_json_str = json.dumps(input_dict)

            # Default prompt
            DEFAULT_PROMPT = """
                Here you are given some information about an interactable element, and several possible objects which might have functional connections with it.
                As an expert analyzing the indoor functionalities, you should judge the confidence for each possible connection and provide the reason of your judgment.
                The input information for you is formatted as:  'inter_ele_idx': the interactable element id; 'inter_ele_center': the position of  the interactable element;
                'inter_ele_name': the name of the interactable element; 'obj_idxs': possibly connected objects' ids; 'obj_centers': possibly connected objects' positions;
                'obj_names': possibly connected objects' names; 'captions': the list of descriptions respectively illustrating the functional connections of the interactable element with each object.
                The rules to judge the confidence for each possible connection you should use are as follows:
                1. You should compare each of the item in 'captions' to judge the rank of the possibility for all functional connections.
                Connection with topper rank should be set with higher confidence score.
                2. You should understand the content of each item in 'captions'. If it says the connection seems less possible, the confidence score should be lower. Otherwise it should be higher.
                3. You should combine 'captions' with the spatial locations of the interactable element and the objects for the judgment.
                4. The confidence score must be a number from 0 (larger not equal to) to 1 (smaller not equal to).
                Please output a JSON string (and no others) with the fields 
                "confidence": the list of confidence scores you predict, with each item indicating the confidence of each connection, with the length same as 'obj_idxs';
                "reason": the list of reasons why you make the judgment, with the same length as 'confidence'.
            """

            # chat_completion = client.completions.create(
            #     # model="gpt-3.5-turbo",
            #     model="gpt-4",
            #     prompt=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
            #     timeout=60,  # Timeout in seconds
            # )
            resp_text = _ask_gpt4_text(
                [{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                60,
            )


            output_dict = input_dict
            # try:
            #     # Attempt to parse the output as a JSON
            #     chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
            #     # If the output is a valid JSON, then add it to the output dictionary
            #     output_dict["confidence"] = chat_output_json["confidence"]
            #     output_dict["reason"] = chat_output_json["reason"]
            # except:
            #     output_dict["confidence"] = "FAIL"
            #     output_dict["reason"] = "FAIL"
            #     exit(1)
            try:
                chat_output_json = json.loads(resp_text)
                output_dict["confidence"] = chat_output_json["confidence"]
                output_dict["reason"] = chat_output_json["reason"]
            except:
                output_dict["confidence"] = "FAIL"
                output_dict["reason"] = "FAIL"
                exit(1)

            # Saving the output
            print("Saving pruning graph to file...")
            with open(Path(savedir_debug) / ("prune_graph_" + str(inter_ele_idx) +".json"), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            output_dict = json.load(open(Path(savedir_debug) / ("prune_graph_" + str(inter_ele_idx) +".json"), "r"))
        for edge in func_edges:
            if edge[0] == output_dict["inter_ele_idx"]:
                for i, obj_idx in enumerate(output_dict["obj_idxs"]):
                    if edge[2] == obj_idx:
                        edge[4] *= output_dict["confidence"][i]
        
    for obj_idx, inter_ele_idxs in phase_3_dict.items():
        obj = scene_map[obj_idx]
        conf = np.array(obj['conf'])
        sem_score = []
        for idx in range(len(obj['class_name'])):
            obj_class = obj['class_name'][idx]
            if obj_class in obj['refined_obj_tag']:
                sem_score.append(1.0)
            else:
                sem_score.append(0.6)
        idx_most_conf = np.argsort(conf*np.array(sem_score))[::-1]
        idx_det = idx_most_conf[0]
        obj_image = Image.open(obj["color_path"][idx_det]).convert("RGB")
        if camera_axis == 'Left':
            obj_image = obj_image.rotate(-90, expand=True)
        xyxy = obj["xyxy"][idx_det]
        mask = obj["mask"][idx_det]
        x1, y1, x2, y2 = xyxy
        padding = 30
        obj_image_crop, mask_crop = crop_image_and_mask(obj_image, mask, x1, y1, x2, y2, padding=padding)
        obj_image_crop_modified = obj_image_crop  # No modification

        captions_list = []
        inter_ele_names_list = []
        for inter_ele_idx in inter_ele_idxs:
            inter_ele = scene_map[inter_ele_idx]
            conf = np.array(inter_ele['conf'])
            inter_ele_names_list.append(inter_ele['refined_obj_tag'])
            sem_score = []
            for idx in range(len(inter_ele['class_name'])):
                inter_ele_class = inter_ele['class_name'][idx]
                if 'remote' in inter_ele_class:
                    sem_score.append(1.0)
                else:
                    sem_score.append(0.6)
            idx_most_conf = np.argsort(conf*np.array(sem_score))[::-1]
            idx_det = idx_most_conf[0]
            image = Image.open(inter_ele["color_path"][idx_det]).convert("RGB")
            if camera_axis == 'Left':
                image = image.rotate(-90, expand=True)
            xyxy = inter_ele["xyxy"][idx_det]
            mask = inter_ele["mask"][idx_det]
            x1, y1, x2, y2 = xyxy
            padding = 30
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            image_crop_modified = image_crop
            concat_image = concatenate_images_horizontal([image_crop_modified, obj_image_crop_modified], 20)
            query = '''
                You are given two images. The left image depicts a
            ''' + inter_ele['refined_obj_tag'] +\
            '''
                , while the right shows a 
            ''' + obj['refined_obj_tag'] +\
            '''
                . Please illustrate if the 
            ''' + inter_ele['refined_obj_tag'] +\
            '''
                 in the left image controls the  
            ''' + obj['refined_obj_tag'] +\
            '''
                 in the right image. You should provide your illustration, reason, and the confidence score between 0 to 1.
                 Format your answer with: Illustration: ...  Confidence: ... Reason: ...
            '''
            console.print("[bold red]User:[/bold red] " + query)
            outputs = chat.infer(
                query = query,
                images = [concat_image],
            )
            console.print("[bold green]LLaVA:[/bold green] " + outputs)
            captions_list.append(outputs)
        
        if not (Path(savedir_debug) / ("prune_graph_" + str(obj_idx) +".json")).exists():
            input_dict = {
                'obj_idx': obj_idx,
                'obj_name': obj['refined_obj_tag'],
                'inter_ele_idxs': inter_ele_idxs,
                'inter_ele_names': inter_ele_names_list,
                'captions': captions_list
            }
            input_json_str = json.dumps(input_dict)

            # Default prompt
            DEFAULT_PROMPT = """
                Here you are given some information about an object, and several possible interactable elements which might have functional connections with it.
                As an expert analyzing the indoor functionalities, you should judge the confidence for each possible connection and provide the reason of your judgement.
                The input information for you is formated as:  'obj_idx': the object id; 
                'obj_name': the name of the object; 'inter_ele_idxs': possible interactable elements' ids;
                'inter_ele_names': possible interactable elements' names; 'captions': the list of descriptions respectively illustrating the functional connections of the object with each interactable element.
                The rules to judge the confidence for each possible connection you should use are as follows:
                1. You should compare each of the item in 'captions' to judge the rank of the possibility for all functional connections.
                Connection with topper rank should be set with higher confidence score.
                2. You should understand the content of each item in 'captions'. If it says the connection seems less possible, the confidence score should be lower. Otherwise it should be higher.
                3. The confidence score must be a number from 0 (larger not equal to) to 1 (smaller not equal to).
                Please output a JSON string (and no others) with the fields 
                "confidence": the list of confidence scores you predict, with each item indicating the confidence of each connection, with the length same as 'obj_idxs';
                "reason": the list of reasons why you make the judgement, with the same length as 'confidence'.
            """

            chat_completion = client.completions.create(
                # model="gpt-3.5-turbo",
                model="gpt-4",
                prompt=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                timeout=60,  # Timeout in seconds
            )

            output_dict = input_dict
            try:
                # Attempt to parse the output as a JSON
                chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
                # If the output is a valid JSON, then add it to the output dictionary
                output_dict["confidence"] = chat_output_json["confidence"]
                output_dict["reason"] = chat_output_json["reason"]
            except:
                output_dict["confidence"] = "FAIL"
                output_dict["reason"] = "FAIL"
                exit(1)
            # Saving the output
            print("Saving pruning graph to file...")
            with open(Path(savedir_debug) / ("prune_graph_" + str(obj_idx) +".json"), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            output_dict = json.load(open(Path(savedir_debug) / ("prune_graph_" + str(obj_idx) +".json"), "r"))
        for edge in func_edges:
            if edge[2] == output_dict["obj_idx"]:
                for i, inter_ele_idx in enumerate(output_dict["inter_ele_idxs"]):
                    if edge[0] == obj_idx:
                        if len(edge) == 5:
                            edge[4] *= output_dict["confidence"][i]
                        elif len(edge) == 4:
                            edge.append(output_dict["confidence"][i])
    edge_file = Path(args.cachedir).parent / "cfslam_funcgraph_edges.pkl"
    with open(Path(edge_file.split('.')[-2]+'_confidence.pkl'), "wb") as f:
        pkl.dump(func_edges, f)
        print('Save edges with confidence!')


def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)
    with gzip.open(args.mapfile, "rb") as f:
        results = pkl.load(f)
    with gzip.open(args.part_file, "rb") as f:
        part_results = pkl.load(f)

    # rigid alignment
    args.cachedir = args.dataset_root + '/' + args.scene_name + '/rigid'
    args.llava_mode = 'rigid'
    extract_node_captions(args, results)

    refine_node_captions(args)

    results, part_results = filter_rigid_objects(args, results, part_results)
    
    args.masking_option = 'red_outline'
    extract_part_captions(args, results, part_results) 

    args.part_reg = True
    refine_node_captions(args)

    results, part_results, scenegraph_edges = build_rigid_funcgraph(args, results, part_results)

    # remote alignment 
    args.cachedir = args.dataset_root + '/' + args.scene_name + '/others'
    args.llava_mode = 'none'
    args.masking_option = 'none'
    extract_node_captions(args, results)

    args.part_reg = False
    refine_node_captions(args)

    results = filter_remote_interactable_objects(args, results)  # store results

    scenegraph_edges = build_object_funcgraph(args, results, scenegraph_edges)     # store edges
    
    prune_graph(args, results, scenegraph_edges)   # store edges

if __name__ == "__main__":
    main()
