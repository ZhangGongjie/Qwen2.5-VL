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

# Omni3D - 3D Object Detection
OMNI3D_KITTI_TRAIN_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/KITTI_train_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_KITTI_VAL_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/KITTI_val_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_KITTI_TEST_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/KITTI_test_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_NUSCENES_TRAIN_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/nuScenes_train_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_NUSCENES_VAL_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/nuScenes_val_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_NUSCENES_TEST_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/nuScenes_test_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_HYPERSIM_TRAIN_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/Hypersim_train_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_HYPERSIM_VAL_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/Hypersim_val_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_HYPERSIM_TEST_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/Hypersim_test_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_ARKITSCENES_TRAIN_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/ARKitScenes_train_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_ARKITSCENES_VAL_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/ARKitScenes_val_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_ARKITSCENES_TEST_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/ARKitScenes_test_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_OBJECTRON_TRAIN_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/Objectron_train_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_OBJECTRON_VAL_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/Objectron_val_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_OBJECTRON_TEST_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/Objectron_test_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_SUNRGBD_TRAIN_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/SUNRGBD_train_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_SUNRGBD_VAL_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/SUNRGBD_val_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}

OMNI3D_SUNRGBD_TEST_3D_OBJECT_DETECTION_CAMCOORD = {
    "annotation_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo/omni3d_3dod_qa/SUNRGBD_test_qa.json",
    "data_path": "/mnt/gongjie_NAS2/Datasets/Omni3D/omni3d_github_repo",
}






data_dict = {
    # ------------ 2D Datasets ------------

    # COCO -- llava
    "coco_complex_reasoning_77k": COCO_COMPLEX_REASONING_77K,
    "coco_conversation_58k": COCO_CONVERSATION_58K,
    "coco_detail_23k": COCO_DETAIL_23K,

    # ------------ 3D Datasets ------------
    # 3D Object Detection
    #  - Omni3D

    "omni3d_nuscenes_train_3d_object_detection_under_cam_coordsys": OMNI3D_NUSCENES_TRAIN_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_nuscenes_val_3d_object_detection_under_cam_coordsys": OMNI3D_NUSCENES_VAL_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_nuscenes_test_3d_object_detection_under_cam_coordsys": OMNI3D_NUSCENES_TEST_3D_OBJECT_DETECTION_CAMCOORD,

    "omni3d_kitti_train_3d_object_detection_under_cam_coordsys": OMNI3D_KITTI_TRAIN_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_kitti_val_3d_object_detection_under_cam_coordsys": OMNI3D_KITTI_VAL_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_kitti_test_3d_object_detection_under_cam_coordsys": OMNI3D_KITTI_TEST_3D_OBJECT_DETECTION_CAMCOORD,

    "omni3d_sunrgbd_train_3d_object_detection_under_cam_coordsys": OMNI3D_SUNRGBD_TRAIN_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_sunrgbd_val_3d_object_detection_under_cam_coordsys": OMNI3D_SUNRGBD_VAL_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_sunrgbd_test_3d_object_detection_under_cam_coordsys": OMNI3D_SUNRGBD_TEST_3D_OBJECT_DETECTION_CAMCOORD,

    "omni3d_hypersim_train_3d_object_detection_under_cam_coordsys": OMNI3D_HYPERSIM_TRAIN_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_hypersim_val_3d_object_detection_under_cam_coordsys": OMNI3D_HYPERSIM_VAL_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_hypersim_test_3d_object_detection_under_cam_coordsys": OMNI3D_HYPERSIM_TEST_3D_OBJECT_DETECTION_CAMCOORD,

    "omni3d_arkitscenes_train_3d_object_detection_under_cam_coordsys": OMNI3D_ARKITSCENES_TRAIN_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_arkitscenes_val_3d_object_detection_under_cam_coordsys": OMNI3D_ARKITSCENES_VAL_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_arkitscenes_test_3d_object_detection_under_cam_coordsys": OMNI3D_ARKITSCENES_TEST_3D_OBJECT_DETECTION_CAMCOORD,

    "omni3d_objectron_train_3d_object_detection_under_cam_coordsys": OMNI3D_OBJECTRON_TRAIN_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_objectron_val_3d_object_detection_under_cam_coordsys": OMNI3D_OBJECTRON_VAL_3D_OBJECT_DETECTION_CAMCOORD,
    "omni3d_objectron_test_3d_object_detection_under_cam_coordsys": OMNI3D_OBJECTRON_TEST_3D_OBJECT_DETECTION_CAMCOORD,

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
