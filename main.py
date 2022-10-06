#!/usr/bin/env python3

import os
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import argparse
from yolov4.common import (
    yolo_tpu_layer,
    get_yolo_tiny_detections,
    fit_to_original
)
from yolov4.common.media import draw_bboxes, resize_image

quant_model_path = "quant_yolov4.tflite"
edgetpu_model_path = "quant_yolov4_edgetpu.tflite"
compile_commands = {
    "partial-tpu-1": f"edgetpu_compiler {quant_model_path} "
                     "-i yolov4-tiny/route_10/concat",
    "partial-tpu-2": f"edgetpu_compiler {quant_model_path} "
                     "-i \"yolov4-tiny/convolutional_19/batch_normalization_18"
                     "/FusedBatchNormV3;yolov4-tiny/convolutional_10/conv2d_10"
                     "/Conv2D;yolov4-tiny/convolutional_19/conv2d_19/Conv2D1\"",
    "full-tpu": f"edgetpu_compiler {quant_model_path}"
}
names_path = "coco.names"
demo_img_path = "city.png"


def convert(config):
    if config in compile_commands:
        os.system(compile_commands[config])
        return edgetpu_model_path
    return quant_model_path


def load_label_names():
    with open(names_path, "r") as infile:
        names = infile.read().split("\n")
    return names


def get_input_img(img, input_details):
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = resize_image(input_img, (416, 416))
    scale, zero_point = input_details[0]["quantization"]
    input_img = input_img / 255.0
    input_img = (input_img / scale + zero_point).astype(np.uint8)
    input_img = input_img[np.newaxis, :, :, :]
    return input_img


def inference(input_img, interpreter, input_details, output_details):
    interpreter.set_tensor(input_details[0]["index"], input_img)
    interpreter.invoke()
    quant_yolos = [
        interpreter.get_tensor(output_detail["index"])
        for output_detail in output_details
    ]
    yolos = []
    for yolo, output_detail in zip(
            reversed(quant_yolos),
            reversed(output_details)):
        yolo = yolo.astype(np.float32)
        scale, zero_point = output_detail["quantization"]
        yolo = ((yolo - float(zero_point)) * scale).astype(np.float32)
        yolos.append(yolo)
    for i in range(2):
        yolo_tpu_layer(yolos[2 * i], yolos[2 * i + 1], 3, 1.05)
    return [yolos[1], yolos[3]]


def run(model_path, names):
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate("libedgetpu.so.1", {})]
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.imread(demo_img_path)
    input_img = get_input_img(img, input_details)
    output = inference(input_img, interpreter, input_details, output_details)

    detections = get_yolo_tiny_detections(
        yolo_0=output[0],
        yolo_1=output[1],
        mask_0=np.array([4, 5, 6], dtype=np.int32),
        mask_1=np.array([1, 2, 3], dtype=np.int32),
        anchors=np.array(
            [[2.403846196830272675e-02, 3.365384787321090698e-02],
             [5.528846010565757751e-02, 6.490384787321090698e-02],
             [8.894230425357818604e-02, 1.394230723381042480e-01],
             [1.947115361690521240e-01, 1.971153914928436279e-01],
             [3.245192170143127441e-01, 4.062500000000000000e-01],
             [8.269230723381042480e-01, 7.668269276618957520e-01]],
            dtype=np.float32),
        beta_nms=0.6,
        new_coords=False,
        prob_thresh=0.3
    )
    fit_to_original(detections, 416, 416, img.shape[0], img.shape[1])
    detection_img = draw_bboxes(img, detections, names=names)
    cv2.imshow(demo_img_path, detection_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_config(g, name, help_str):
    g.add_argument(
        f"--{name}",
        dest="config",
        action="store_const",
        const=name,
        help=help_str
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference on CPU or Edge TPU."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    add_config(g, "full-cpu", "run entire model on the CPU")
    add_config(
        g,
        "partial-tpu-1",
        "run all operations before (not including) conv2d on the TPU"
    )
    add_config(
        g,
        "partial-tpu-2",
        "run all operations before (including) conv2d on the TPU"
    )
    add_config(g, "full-tpu", "run entire model on the TPU")
    args = parser.parse_args()
    return args.config


def main(config):
    model_path = convert(config)
    names = load_label_names()
    run(model_path, names)


if __name__ == "__main__":
    main(parse_arguments())
