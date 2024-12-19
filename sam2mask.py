import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Environment setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def load_image_frames(image_dir):
    frame_names = [
        p for p in os.listdir(image_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names

def extract_frames_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            print(f"Skipping empty or corrupted frame at index {frame_idx}")
            continue
        frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        if not os.path.exists(frame_path):
            print(f"Failed to save frame at index {frame_idx}")
        frame_idx += 1
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")

def save_segmentation_result(frame_path, mask, output_path):
    if mask is None or mask.size == 0:
        print(f"Warning: Empty mask for frame {frame_path}")
        mask = np.zeros((566, 1008), dtype=np.uint8) 
    else:
        mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask)
    if not os.path.exists(output_path):
        print(f"Failed to save mask for frame {frame_path}")

def select_frame_gui(image_dir):
    frame_names = load_image_frames(image_dir)
    fig, ax = plt.subplots()
    frame_idx = 0

    def update_frame(idx):
        frame_path = os.path.join(image_dir, frame_names[idx])
        img = Image.open(frame_path)
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Frame {idx}")
        plt.draw()

    def on_key(event):
        nonlocal frame_idx
        if event.key == 'right' and frame_idx < len(frame_names) - 1:
            frame_idx += 1
        elif event.key == 'left' and frame_idx > 0:
            frame_idx -= 1
        elif event.key == 'enter':
            plt.close()
        update_frame(frame_idx)

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_frame(frame_idx)
    plt.show()

    return frame_idx

def select_point_on_frame(frame_path):
    fig, ax = plt.subplots()
    img = Image.open(frame_path)
    ax.imshow(img)
    points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click on the frame to select a point")
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if not points:
        raise ValueError("No points selected")

    return np.array(points, dtype=np.float32)

def process_images_with_tracking(image_dir, output_dir, predictor, inference_state, points, start_frame_idx=0, ann_obj_id=1):
    frame_names = load_image_frames(image_dir)

    labels = np.array([1] * len(points), np.int32)
    _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=start_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    for idx in range(start_frame_idx):
        frame_path = os.path.join(image_dir, frame_names[idx])
        output_path = os.path.join(mask_dir, f"mask_{idx:05d}.png")
        black_mask = np.zeros((566, 1008), dtype=np.uint8)
        save_segmentation_result(frame_path, black_mask, output_path)

    for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if frame_idx < start_frame_idx:
            continue

        for i, obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            frame_path = os.path.join(image_dir, frame_names[frame_idx])
            if not os.path.exists(frame_path):
                print(f"Warning: Frame {frame_path} does not exist")
                continue
            output_path = os.path.join(mask_dir, f"mask_{frame_idx:05d}.png")
            save_segmentation_result(frame_path, mask, output_path)

    print(f"Saved segmentation results to {mask_dir}")

def process_video_with_tracking(video_dir, output_dir, predictor, inference_state, points, start_frame_idx=0, ann_obj_id=1):
    frame_names = load_image_frames(video_dir)

    labels = np.array([1] * len(points), np.int32)
    _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=start_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    for idx in range(start_frame_idx):
        frame_path = os.path.join(video_dir, frame_names[idx])
        output_path = os.path.join(mask_dir, f"mask_{idx:05d}.png")
        black_mask = np.zeros((566, 1008), dtype=np.uint8)
        save_segmentation_result(frame_path, black_mask, output_path)

    for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if frame_idx < start_frame_idx:
            continue

        for i, obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            frame_path = os.path.join(video_dir, frame_names[frame_idx])
            if not os.path.exists(frame_path):
                print(f"Warning: Frame {frame_path} does not exist")
                continue
            output_path = os.path.join(mask_dir, f"mask_{frame_idx:05d}.png")
            save_segmentation_result(frame_path, mask, output_path)

    print(f"Saved segmentation results to {mask_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process video or image folder with SAM 2 segmentation.")
    parser.add_argument("--input_video", type=str, help="Path to the input video.")
    parser.add_argument("--input_images", type=str, help="Path to the input image folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    args = parser.parse_args()

    if not args.input_video and not args.input_images:
        raise ValueError("Either --input_video or --input_images must be provided.")

    device = setup_device()

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    if args.input_video:
        video_dir = os.path.join(args.output_dir, "frames")
        extract_frames_from_video(args.input_video, video_dir)

        inference_state = predictor.init_state(video_path=video_dir)

        start_frame_idx = select_frame_gui(video_dir)
        frame_path = os.path.join(video_dir, load_image_frames(video_dir)[start_frame_idx])
        points = select_point_on_frame(frame_path)

        process_video_with_tracking(video_dir, args.output_dir, predictor, inference_state, points, start_frame_idx=start_frame_idx)

    elif args.input_images:
        inference_state = predictor.init_state(video_path=args.input_images)

        start_frame_idx = select_frame_gui(args.input_images)
        frame_path = os.path.join(args.input_images, load_image_frames(args.input_images)[start_frame_idx])
        points = select_point_on_frame(frame_path)

        process_images_with_tracking(args.input_images, args.output_dir, predictor, inference_state, points, start_frame_idx=start_frame_idx)

if __name__ == "__main__":
    main()
