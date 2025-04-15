import re

# ================================= 2D =================================

COCO_COMPLEX_REASONING_77K = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/ICCV2025/Official_Projects/LLaVA-NeXT-dev/data/complex_reasoning_77k.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

COCO_CONVERSATION_58K = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/ICCV2025/Official_Projects/LLaVA-NeXT-dev/data/conversation_58k.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

COCO_DETAIL_23K = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/ICCV2025/Official_Projects/LLaVA-NeXT-dev/data/detail_23k.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}




# ================================= 3D =================================

COCO_COMPLEX_REASONING_3D_77K = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/ICCV2025/Official_Projects/LLaVA-NeXT-dev/data/complex_reasoning_77k_3D.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

COCO_CONVERSATION_3D_58K = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/ICCV2025/Official_Projects/LLaVA-NeXT-dev/data/conversation_58k_3D.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

COCO_DETAIL_3D_23K = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/ICCV2025/Official_Projects/LLaVA-NeXT-dev/data/detail_23k_3D.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

COCO_3DCOORD_GROUNDING = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/coord_understanding_coco3d_3D_image.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}



SCANNET_2D_EMBODIED_DIALOGUE_TRAIN = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_embodied_dialogue_filtered_train.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_EMBODIED_DIALOGUE_VAL = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_embodied_dialogue_filtered_val.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_EMBODIED_PLANNING_TRAIN = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_embodied_planning_filtered_train.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_EMBODIED_PLANNING_VAL = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_embodied_planning_filtered_val.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_EMBODIED_QA_TRAIN = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_embodied_question_answer_train.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_EMBODIED_QA_VAL = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_embodied_question_answer_val.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_ROOM_DESCRIPTION_TRAIN = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_scene_description_train.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_ROOM_DESCRIPTION_VAL = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/modified_3d_llm_scene_description_val.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_3DCOORD_GROUNDING_TRAIN = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/final_scannet_3dcoordQA_train.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}

SCANNET_2D_3DCOORD_GROUNDING_VAL = {
    "annotation_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data/final_scannet_3dcoordQA_val.json",
    "data_path": "/mnt/gongjie_NAS2/CodeSpace/EmbodiedAI_Research/20250410/QWen-3DVL/data"
}






data_dict = {
    # ------------ 2D Datasets ------------

    # COCO -- llava
    "coco_complex_reasoning_77k": COCO_COMPLEX_REASONING_77K,
    "coco_conversation_58k": COCO_CONVERSATION_58K,
    "coco_detail_23k": COCO_DETAIL_23K,

    # ------------ 3D Datasets ------------
    # COCO 3D
    "coco_complex_reasoning_3d_77k": COCO_COMPLEX_REASONING_3D_77K,
    "coco_conversation_3d_58k": COCO_CONVERSATION_3D_58K,
    "coco_detail_3d_23k": COCO_DETAIL_3D_23K,
    "coco_3dcoord_grounding": COCO_3DCOORD_GROUNDING,

    # ScanNet 2D
    "scannet_2d_embodied_dialogue_train": SCANNET_2D_EMBODIED_DIALOGUE_TRAIN,
    "scannet_2d_embodied_dialogue_val": SCANNET_2D_EMBODIED_DIALOGUE_VAL,
    "scannet_2d_embodied_planning_train": SCANNET_2D_EMBODIED_PLANNING_TRAIN,
    "scannet_2d_embodied_planning_val": SCANNET_2D_EMBODIED_PLANNING_VAL,
    "scannet_2d_embodied_qa_train": SCANNET_2D_EMBODIED_QA_TRAIN,
    "scannet_2d_embodied_qa_val": SCANNET_2D_EMBODIED_QA_VAL,
    "scannet_2d_room_description_train": SCANNET_2D_ROOM_DESCRIPTION_TRAIN,
    "scannet_2d_room_description_val": SCANNET_2D_ROOM_DESCRIPTION_VAL,
    "scannet_2d_3dcoord_grounding_train": SCANNET_2D_3DCOORD_GROUNDING_TRAIN,
    "scannet_2d_3dcoord_grounding_val": SCANNET_2D_3DCOORD_GROUNDING_VAL,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["coco_complex_reasoning_77k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
