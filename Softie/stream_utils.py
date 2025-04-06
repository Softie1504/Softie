# import cv2
# import pyvirtualcam
# import torch
# import numpy as np

# from engine import ClusterSegmentationWithYolo

# class Streaming(ClusterSegmentationWithYolo):
#     def __init__(self, in_source=None, out_source=None, fps=None, blur_strength=None, cam_fps=15, background="none"):
#         super().__init__(erode_size=5, erode_intensity=2)
#         self.input_source = in_source
#         self.output_source = out_source
#         self.fps = fps
#         self.blur_strength = blur_strength
#         self.background = background
#         self.running = False
#         self.original_fps = fps
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#         print(f"Device selected/found for inference : {self.device}")

#     def update_streaming_config(self, in_source=None, out_source=None, fps=None, blur_strength=None, background="none"):
#         self.input_source = in_source
#         self.output_source = out_source
#         self.fps = fps
#         self.blur_strength = blur_strength
#         self.background = background

#     def update_cam_fps(self, fps):
#         self.original_fps = fps

#     def update_running_status(self, running_status=False):
#         self.running = running_status


#     def stream_video(self):
#         self.running = True
#         print(f"Retreiving feed from source({self.input_source}), FPS : {self.fps}, Blur Strength : {self.blur_strength}")
#         cap = cv2.VideoCapture(int(self.input_source), cv2.CAP_DSHOW) ##
 
#         frame_idx = 0

#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         try:
#             self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
#         except Exception as e:
#             print(f"Webcam({self.input_source}) live fps is not available. Set the fps accordingly. Exception: {e}")

#         if self.fps:
#             if self.fps > self.original_fps:
#                 self.fps = self.original_fps
#                 frame_interval = int(self.original_fps / self.fps) 
#             else:
#                 frame_interval = int(self.original_fps / self.fps)
#         else:
#             frame_interval = 1

#     # #  FIX: Avoid ZeroDivisionError by checking fps
#     #     if self.fps and self.fps > 0:
#     #         if self.fps > self.original_fps:
#     #             self.fps = self.original_fps
#     #         frame_interval = max(1, int(self.original_fps / self.fps))
#     #     else:
#     #         print("FPS must be greater than 0. Setting default frame_interval = 1.")
#     #         frame_interval = 1
#     #         self.fps = self.original_fps

#         with pyvirtualcam.Camera(width=width, height=height, fps=self.fps) as cam:
#             print(f"Virtual Camera running at {width}x{height} as {self.fps}")

#             while self.running and cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 if frame_idx % frame_interval == 0:
#                     results = self.model.predict(source=frame, save=False, save_txt=False, stream=True, retina_masks=True, verbose=False, device=self.device)  # retina_mask=True was valid in YOLOv5
#                     mask = self.genrate_mask_from_result(results)

#                     if mask is not None:
#                         if self.background == "blur":
#                             result_frame = self.apply_blur_with_mask(frame, mask, blur_strength=self.blur_strength)
#                         elif self.background == "none":
#                             result_frame = self.apply_black_background(frame, mask)
#                         elif self.background == "default":
#                             result_frame = self.apply_custom_background(frame, mask)
#                     #     else:
#                     #         result_frame = frame  # fallback in case of unknown background
#                     # else:
#                     #     result_frame = frame  # fallback if mask is None

#                     # # process mask and create result
#                 #     result_frame = 0

#                 # frame_idx += 1

#             cam.send(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
#             cam.sleep_until_next_frame()

#         cap.release()

#     def list_available_devices(self):
#         devices = []
#         for i in range(5):
#             cap = cv2.VideoCapture(i)
#             if cap.isOpened():
#                 devices.append({"id": i, "name": f"Camera {i}"})
#                 cap.release()
#         return devices
    
#     if __name__ == "__main__":
#         print(list_available_devices())

import cv2
import pyvirtualcam
import torch
import numpy as np

from engine import ClusterSegmentationWithYolo

class Streaming(ClusterSegmentationWithYolo):
    def __init__(self, in_source=None, out_source=None, fps=None, blur_strength=None, cam_fps=15, background="none"):
        super().__init__(erode_size=5, erode_intensity=2)
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.background = background
        self.running = False
        self.original_fps = fps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device selected/found for inference : {self.device}")

    def update_streaming_config(self, in_source=None, out_source=None, fps=None, blur_strength=None, background="none"):
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.background = background

    def update_cam_fps(self, fps):
        self.original_fps = fps

    def update_running_status(self, running_status=False):
        self.running = running_status

    def stream_video(self):
        self.running = True
        print(f"Retreiving feed from source({self.input_source}), FPS : {self.fps}, Blur Strength : {self.blur_strength}")
        cap = cv2.VideoCapture(int(self.input_source), cv2.CAP_DSHOW)
        frame_idx = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if self.original_fps == 0:
                raise ValueError("FPS returned 0")
        except Exception as e:
            print(f"Webcam({self.input_source}) live fps is not available. Set the fps accordingly. Exception: {e}")
            self.original_fps = 30  # default fallback

        # ✅ Safe FPS and interval check
        if self.fps and self.fps > 0:
            if self.fps > self.original_fps:
                self.fps = self.original_fps
            frame_interval = max(1, int(self.original_fps / self.fps))
        else:
            print("FPS not set or invalid. Using default values.")
            self.fps = self.original_fps
            frame_interval = 1

        with pyvirtualcam.Camera(width=width, height=height, fps=self.fps) as cam:
            print(f"Virtual Camera running at {width}x{height} as {self.fps} FPS")

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    results = self.model.predict(
                        source=frame,
                        save=False,
                        save_txt=False,
                        stream=True,
                        retina_masks=True,
                        verbose=False,
                        device=self.device
                    )

                    mask = self.genrate_mask_from_result(results)

                    if mask is not None:
                        if self.background == "blur":
                            print(self.background)
                            result_frame = self.apply_blur_with_mask(frame, mask, blur_strength=self.blur_strength)
                        elif self.background == "none":
                            print(self.background)
                            result_frame = self.apply_black_background(frame, mask)
                        elif self.background == "default":
                            print(self.background)
                            result_frame = self.apply_custom_background(frame, mask)
                        else:
                            result_frame = frame
                    else:
                        result_frame = frame

                    cam.send(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()

                frame_idx += 1

        cap.release()

    @staticmethod
    def list_available_devices():
        devices = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({"id": i, "name": f"Camera {i}"})
                cap.release()
        return devices


if __name__ == "__main__":
    print(Streaming.list_available_devices())
