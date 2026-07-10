# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import onnx
import torch
import argparse
import numpy as np

from pathlib import Path
from onnxsim import simplify
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file

from modify_onnx import simplify_preprocess, simplify_postprocess

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--out_dir', type=str, default='model', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def compute_export_shapes(cfg):
    """Derive the BEV grid / anchor-head channel layout from a cfg so the
    ONNX graph surgery matches models with a different point cloud range,
    voxel size, or class/anchor count than KITTI's 3-class default.
    """
    pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float64)
    voxel_size = np.array(
        next(d['VOXEL_SIZE'] for d in cfg.DATA_CONFIG.DATA_PROCESSOR
             if d['NAME'] == 'transform_points_to_voxels'),
        dtype=np.float64)
    grid_size = np.round((pc_range[3:6] - pc_range[0:3]) / voxel_size).astype(int)

    anchor_cfgs = cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG
    feature_map_stride = anchor_cfgs[0].get('feature_map_stride', 2)
    num_anchors = sum(len(a['anchor_rotations']) * len(a['anchor_sizes']) for a in anchor_cfgs)
    num_classes = len(cfg.CLASS_NAMES)
    num_dir_bins = cfg.MODEL.DENSE_HEAD.get('NUM_DIR_BINS', 2)

    feature_w = int(grid_size[0] // feature_map_stride)
    feature_h = int(grid_size[1] // feature_map_stride)
    dense_shape = (int(grid_size[1]), int(grid_size[0]))

    return {
        'feature_h': feature_h,
        'feature_w': feature_w,
        'num_anchors': num_anchors,
        'num_classes': num_classes,
        'num_box_values': 7,
        'num_dir_bins': num_dir_bins,
        'dense_shape': dense_shape,
    }


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('------ Convert OpenPCDet model for TensorRT ------')
    export_shapes = compute_export_shapes(cfg)
    logger.info('Derived export shapes: %s', export_shapes)
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext='.bin', logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)

    model.cuda()
    model.eval()

    np.set_printoptions(threshold=np.inf)

    with torch.no_grad():

      MAX_VOXELS = 10000

      dummy_voxels = torch.zeros(
          (MAX_VOXELS, 32, 4),
          dtype=torch.float32,
          device='cuda:0')

      dummy_voxel_idxs = torch.zeros(
          (MAX_VOXELS, 4),
          dtype=torch.int32,
          device='cuda:0')

      dummy_voxel_num = torch.zeros(
          (1,),
          dtype=torch.int32,
          device='cuda:0')

      dummy_input = dict()
      dummy_input['voxels'] = dummy_voxels
      dummy_input['voxel_num_points'] = dummy_voxel_num
      dummy_input['voxel_coords'] = dummy_voxel_idxs
      dummy_input['batch_size'] = 1

      torch.onnx.export(model,       # model being run
          dummy_input,               # model input (or a tuple for multiple inputs)
          os.path.join(args.out_dir, "pointpillar_raw.onnx"),  # where to save the model (can be a file or file-like object)
          export_params=True,        # store the trained parameter weights inside the model file
          opset_version=11,          # the ONNX version to export the model to
          do_constant_folding=True,  # whether to execute constant folding for optimization
          keep_initializers_as_inputs=True,
          input_names = ['voxels', 'voxel_num', 'voxel_idxs'],   # the model's input names
          output_names = ['cls_preds', 'box_preds', 'dir_cls_preds'], # the model's output names
          )

    onnx_raw = onnx.load(os.path.join(args.out_dir, "pointpillar_raw.onnx"))  # load onnx model
    onnx_trim_post = simplify_postprocess(
        onnx_raw,
        feature_h=export_shapes['feature_h'],
        feature_w=export_shapes['feature_w'],
        num_anchors=export_shapes['num_anchors'],
        num_classes=export_shapes['num_classes'],
        num_box_values=export_shapes['num_box_values'],
        num_dir_bins=export_shapes['num_dir_bins'],
    )

    onnx_simp, check = simplify(onnx_trim_post)
    assert check, "Simplified ONNX model could not be validated"

    onnx_final = simplify_preprocess(onnx_simp, dense_shape=export_shapes['dense_shape'])
    onnx.save(onnx_final, os.path.join(args.out_dir, "pointpillar.onnx"))

    logger.info('[PASS] ONNX EXPORTED.')

if __name__ == '__main__':
    main()
