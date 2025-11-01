# Jumbled Frames Reconstruction

## Objective
Reconstruct a 10-second jumbled video (300 frames) into the correct sequential order.

1. Installation
```bash
pip install -r requirements.txt
```
2. add the video to the directory from device


3. Basic run
```bash
python main.py
```
4. Specify input and output files
```bash
python main.py --input my_video.mp4 --output reconstructed.mp4
```
or
```bash
python main.py -i input_video.mp4 -o output_video.mp4
```
5. A new folder "frames" will be created to store all extracted frames


6. The reconstructed video will be saved to the directory itself. go to files and play the video in amy media player.


7. The log of the reconstruction is saved in execution_log.txt