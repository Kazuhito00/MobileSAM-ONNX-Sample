#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
from collections import deque

from typing import Any, Tuple, Dict, List, Deque, Optional

import cv2
import numpy as np
import onnxruntime  # type: ignore


def get_args() -> Any:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default='sample.png',
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default='onnx_model/vit_t_encoder.onnx',
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default='onnx_model/vit_t_decoder.onnx',
    )

    args = parser.parse_args()

    return args


mouse_point: List[int] = [0] * 2
click_mode: int
click_info: Dict[int, Deque]


def mouse_callback(
    event: int,
    x: float,
    y: float,
    flags: int,
    param: Any,
) -> None:
    global click_mode, click_info, mouse_point

    mouse_point[0] = int(x)
    mouse_point[1] = int(y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if click_mode == 3:
            if len(click_info[click_mode]) == 0:
                click_info[click_mode].append([int(x), int(y)])
            elif len(click_info[click_mode]) == 2:
                click_info[click_mode].clear()
                click_info[click_mode].append([int(x), int(y)])
        else:
            click_info[click_mode].append([int(x), int(y)])
    elif event == cv2.EVENT_LBUTTONUP:
        if click_mode == 3:
            if len(click_info[click_mode]) == 1:
                click_info[click_mode].append([int(x), int(y)])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if click_mode != 3:
            if len(click_info[click_mode]) > 0:
                click_info[click_mode].pop()
        else:
            click_info[click_mode].clear()


def preprocess_image(
    image: np.ndarray,
    resize_width: int = 1024,
    mean: np.ndarray = np.array([123.675, 116.28, 103.53]),
    std: np.ndarray = np.array([58.395, 57.12, 57.375]),
) -> Tuple[np.ndarray, Dict[str, int]]:
    # BGR -> RGB
    temp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 長辺を最大長（デフォルト：1024）にあわせてリサイズ
    image_width, image_height = temp_image.shape[1], temp_image.shape[0]
    resize_image_width, resize_image_height = image_width, image_height

    if image_width > image_height:
        resize_image_width = resize_width
        resize_image_height = int((resize_width / image_width) * image_height)
    else:
        resize_image_width = resize_width
        resize_image_height = int((resize_width / image_height) * image_width)

    temp_image = cv2.resize(temp_image,
                            (resize_image_width, resize_image_height))

    resize_info = {
        "image_width": image_width,
        "image_height": image_height,
        "resize_image_width": resize_image_width,
        "resize_image_height": resize_image_height,
    }

    # 正規化
    temp_image = (temp_image - mean) / std

    # 入力シェイプにあわせてリシェイプ
    temp_image = temp_image.transpose(2, 0,
                                      1)[None, :, :, :].astype(np.float32)

    # 最大長（デフォルト：1024）にあわせてパディング
    if resize_image_height < resize_image_width:
        temp_image = np.pad(temp_image,
                            ((0, 0), (0, 0),
                             (0, resize_width - resize_image_height), (0, 0)))
    else:
        temp_image = np.pad(temp_image,
                            ((0, 0), (0, 0), (0, 0),
                             (0, resize_width - resize_image_width)))

    return temp_image, resize_info


def preprocess_point(
    input_point: List[List[int]],
    input_label: List[int],
    resize_info: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    temp_input_point = np.array(input_point).astype(np.float32)
    temp_input_label = np.array(input_label).astype(np.float32)

    points = temp_input_point.reshape(-1, 2)
    points[..., 0] = points[..., 0] * (resize_info['resize_image_width'] /
                                       resize_info['image_width'])
    points[..., 1] = points[..., 1] * (resize_info['resize_image_height'] /
                                       resize_info['image_height'])

    labels = temp_input_label.flatten()

    points = points[np.newaxis, :, :]
    labels = labels[np.newaxis, :]

    return points, labels


def clickpoint2inputpoint(
    click_info: Dict,
    resize_info: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    # クリック座標数を確認
    point_count: int = 0
    point_count = len(click_info[1])
    point_count += len(click_info[2])
    if len(click_info[3]) == 2:
        point_count += 1

    input_point, input_label = np.array([]), np.array([])
    if point_count > 0:
        target_points = []
        target_labels = []
        # 入力座標列、ラベル列を生成
        if len(click_info[1]) > 0:
            target_points += list(click_info[1])
            target_labels += [1] * len(click_info[1])
        if len(click_info[2]) > 0:
            target_points += list(click_info[2])
            target_labels += [0] * len(click_info[2])
        if len(click_info[3]) == 2:
            x1, y1 = click_info[3][0][0], click_info[3][0][1]
            x2, y2 = click_info[3][1][0], click_info[3][1][1]
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            target_points += list([[x1, y1], [x2, y2]])
            target_labels += list([2, 3])

        # 座標列へ前処理を実施
        input_point, input_label = preprocess_point(
            input_point=target_points,
            input_label=target_labels,
            resize_info=resize_info,
        )

    return input_point, input_label


def main() -> None:
    global click_mode, click_info, mouse_point

    # 引数解析
    args = get_args()
    image_path = args.image
    encoder_path = args.encoder
    decoder_path = args.decoder

    # 画像読み込み
    image = cv2.imread(image_path)

    # エンコーダーセッション生成
    encoder_model = onnxruntime.InferenceSession(
        encoder_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    # デコーダーセッション生成
    decoder_model = onnxruntime.InferenceSession(
        decoder_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    # エンコーダーで推論して、埋め込みベクトルを取得
    preprocessed_image, resize_info = preprocess_image(image)
    embedding = encoder_model.run(None, {'image': preprocessed_image})[0]

    # 画面準備
    window_name = 'Demo'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)  # type: ignore

    mouse_point = [0, 0]
    click_mode = 1
    click_info = {}
    click_info[1] = deque()
    click_info[2] = deque()
    click_info[3] = deque(maxlen=2)

    while True:

        # クリック座標からモデル入力座標へ変換
        input_point, input_label = clickpoint2inputpoint(
            click_info,
            resize_info,
        )

        masks = None
        if len(input_point) > 0 and len(input_label) > 0:
            # デコーダーで推論してマスクを生成
            masks, _, _ = decoder_model.run(
                None, {
                    "image_embedding":
                    embedding,
                    "point_coords":
                    input_point,
                    "point_labels":
                    input_label,
                    "mask_input":
                    np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input":
                    np.zeros(1, dtype=np.float32),
                    "orig_im_size":
                    np.array([
                        resize_info['image_height'], resize_info['image_width']
                    ],
                             dtype=np.float32),
                })

        debug_image = draw_debug_info(
            image,
            masks,
            mouse_point,
            click_info,
        )

        cv2.imshow(window_name, debug_image)
        key = cv2.waitKey(1)
        if 49 <= key <= 51:
            click_mode = key - 48
        if key == 27:  # ESC
            break


def draw_debug_info(
    image: np.ndarray,
    masks: Optional[np.ndarray],
    mouse_point: List[int],
    click_info: Dict[int, Deque[Tuple[int, int]]],
) -> np.ndarray:
    debug_image = copy.deepcopy(image)

    # マスク画像を半透明緑色でオーバーレイ
    if masks is not None:
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 255, 0)

        mask = masks[0][0]
        mask = (mask > 0).astype('uint8') * 255
        mask = cv2.bitwise_not(mask)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')

        mask_image = np.where(mask, debug_image, bg_image)
        debug_image = cv2.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

    # クリック座標を描画
    for click_type, click_deque in click_info.items():
        if click_type == 1 or click_type == 3:
            color = (255, 0, 0)
        elif click_type == 2:
            color = (0, 0, 255)

        for click_point in click_deque:
            cv2.circle(
                debug_image,
                click_point,
                3,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
    if len(click_info[3]) == 1:
        cv2.rectangle(
            debug_image,
            click_info[3][0],
            mouse_point,
            (255, 0, 0),
            2,
        )
    elif len(click_info[3]) == 2:
        x1, y1 = click_info[3][0][0], click_info[3][0][1]
        x2, y2 = click_info[3][1][0], click_info[3][1][1]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return debug_image


if __name__ == '__main__':
    main()
