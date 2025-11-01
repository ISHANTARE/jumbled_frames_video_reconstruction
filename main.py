import cv2
import numpy as np
import os
import time
import argparse
import shutil


def extract_frames(video_path, output_dir="frames"):
    print("Extracting frames from video...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(path, frame)
        frames.append(path)
        count += 1

    cap.release()
    print(f"Extracted {count} frames")
    return frames


def preprocess_frame(path, target_size=(120, 90)):
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return (blur / 255.0).flatten()
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def compute_frame_differences(frames):
    print("Computing frame differences...")
    n = len(frames)
    diff = np.zeros((n, n))

    print("Preprocessing frames...")
    processed = []
    for i in range(n):
        p = preprocess_frame(frames[i])
        processed.append(p if p is not None else np.zeros(120 * 90))

    print("Calculating frame similarities...")
    for i in range(n):
        if i % 50 == 0:
            print(f"  Processed {i}/{n}")
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((processed[i] - processed[j]) ** 2))
                diff[i, j] = d
    return diff


def find_start_frame(diff):
    sums = np.sum(diff, axis=1)
    return np.argmax(sums)


def reconstruct_sequence(diff, start):
    print("Reconstructing frame sequence...")
    n = diff.shape[0]
    visited = {start}
    seq = [start]
    current = start

    for i in range(n - 1):
        if i % 50 == 0:
            print(f"  Reconstructed {i}/{n - 1}")
        d = diff[current].copy()
        for v in visited:
            d[v] = float('inf')
        nxt = np.argmin(d)
        seq.append(nxt)
        visited.add(nxt)
        current = nxt

    return seq


def create_video_from_sequence(frames, seq, output, fps=30):
    print("Creating output video...")
    sample = cv2.imread(frames[0])
    h, w = sample.shape[:2]
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i, idx in enumerate(seq):
        if i % 50 == 0:
            print(f"  Written {i}/{len(seq)} frames")
        frame = cv2.imread(frames[idx])
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"Video saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Jumbled Frames Reconstruction")
    parser.add_argument("--input", "-i", default="jumbled_video.mp4")
    parser.add_argument("--output", "-o", default="reconstructed_video.mp4")
    args = parser.parse_args()

    print("=== Jumbled Frames Reconstruction ===")
    start_time = time.time()

    try:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found")
            print("Please download the jumbled video and save it as 'jumbled_video.mp4'")
            return 1

        frames = extract_frames(args.input)
        if not frames:
            print("Error: No frames extracted from video")
            return 1

        print(f"Processing {len(frames)} frames...")
        diff = compute_frame_differences(frames)
        start_idx = find_start_frame(diff)
        print(f"Selected start frame: {start_idx}")

        seq = reconstruct_sequence(diff, start_idx)
        create_video_from_sequence(frames, seq, args.output)

        elapsed = time.time() - start_time
        print("Reconstruction complete")
        print(f"Execution Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Output: {args.output}")
        print(f"Frames Processed: {len(frames)}")

        with open("execution_log.txt", "w") as f:
            f.write("=== Jumbled Frames Reconstruction Log ===\n")
            f.write(f"Execution Time: {elapsed:.2f} seconds\n")
            f.write(f"Number of Frames: {len(frames)}\n")
            f.write(f"Start Frame: {start_idx}\n")
            f.write("Algorithm: Frame Difference + Nearest Neighbor\n")
            f.write(f"Input: {args.input}\n")
            f.write(f"Output: {args.output}\n")

        print("Execution log saved to: execution_log.txt")

    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())