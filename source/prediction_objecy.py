   # Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import pytesseract
from datetime import datetime, timedelta
import os

from ultralytics.cfg import get_cfg
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, SETTINGS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
#from pathlib import Path

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""

class ObjectDetection:
    def __init__(self):
        # 0 enter_person
        # 1 person
        # 2 car
        # 3 motorcycle
        # 4 bus
        # 5 truck
        # 6 bicycle
        # 7 enter car
        self.count = np.zeros(8, dtype = int)
        self.prev_count = np.zeros(8, dtype = int)
        self.data_whe= []
        self.data_en_p= []
        self.data_en_c= []
        self.data_ou = []
        self.data_ou_count = []
        self.data_car_count = []
        self.inside_P = []
        self.inside_C = []
        self.outside = []
        self.ignore_IN = []
        self.ignore_OUT = []
        self.ignore_IO = []
        self.time = None
        self.time_region = []
        self.date = None
        self.person_conf = 0.8
        self.car_conf = 0.85
        self.out_conf = 0.85
        self.vid_stride = None
        self.display_ratio = 1
        self.df = pd.DataFrame(columns = ['Timestamp','Person', 'Enter Person', 'Car', 'Enter Car', 'Motorcycle', 'Bus', 'Truck', 'Bicycle'])
        self.xlx_path = None

    def compare_index(self, p1, p2):
        x_point = (p1[0], p2[0]) if p1[0] <p2[0] else (p1[0], p2[0])
        y_point = (p1[1], p2[1]) if p1[1] <p2[1] else (p1[1], p2[1])
        return (x_point, y_point)
    
    def check_conf(self, cls, conf):
        if (cls == 0) or cls == 2:
            stan_conf = self.person_conf if (cls == 0) else self.car_conf
        else:
            stan_conf = self.out_conf
        return True if (conf >= stan_conf) else False
    
    def count_cls(self, cls):
        # bicycle
        if cls == 1:
            self.count[6]+=1
        # car
        #elif cls == 2:
        #    self.count[2]+=1
        # motorcycle
        elif cls == 3:
            self.count[3]+=1
        # bus
        elif cls == 5:
            self.count[4]+=1
        # truck
        elif cls == 7:
            self.count[5]+=1
                
    
    # 0 ignore in
    # 1 ignore out
    # 2 ignore in out
    # 3 inside person
    # 4 outside
    # 5 inside car
    def check_intersect(self, region, pt):
        coordinate = None
        x = pt[0]
        y = pt[1]
        if region == 0:
            coordinate = self.ignore_IN        
        elif region == 1:
            coordinate = self.ignore_OUT
        elif region == 2:
            coordinate = self.ignore_IO 
        elif region == 3:
            coordinate = self.inside_P
        elif region == 4:   
            coordinate = self.outside
        elif region == 5:   
            coordinate = self.inside_C
            
        for up_corner, down_corner in coordinate:
            if (x in range(up_corner[0] ,down_corner[0])) and (y in range(up_corner[1],down_corner[1])):
                return True
        return False
    
    def define_error(self, lis):
        for i in lis:
            print(int(i), end=" ")
        print()

    def count_obj(self, centre, cls, id, conf, cam_error):
#        print("BEFORE\n-----------------------------------")
#        print("Enter list: ", self.data_en_p)
#        print("Outside list: ", self.data_ou)
#        print("Wheel list: ", self.data_whe)
#        enter_point = self.compare_index(enter[0], enter[1])
#        out_point = self.compare_index(out[0], out[1])
        self.check_space()
        for i, pts in enumerate(centre):
            ignore_in = False
            ignore_out = False

            # ignore IO
            if(self.check_intersect(2, pts)):
                continue
            # ignore IN
            ignore_in =  self.check_intersect(0, pts)
            # ignore OUT
            ignore_out =  self.check_intersect(1, pts)
            #print("before: ")
            #self.define_error(self.data_ou)
            # inside person
            if self.check_intersect(3, pts) and not ignore_in:
                #print("Current id: ", id[i])
                #print("List enter: ", id[i] in self.data_en_p)
                #print("List out: ", id[i] in self.data_ou)
                #print("list out count: ", id[i] in self.data_ou_count)
                if (id[i] not in self.data_en_p) and (cls[i] == 0):
                    #print("human and not in data en")
                    #cv2.waitKey(0)
                    if (id[i] in self.data_ou):
                       # print(" in data out")
                        self.count[0] += 1 
                        #cv2.waitKey(0)
                        if id[i] not in self.data_ou_count:
                     #       print(" not in data out count")
                            self.count[1]+=1 
                            self.data_ou_count.append(id[i])
                    self.data_en_p.append(id[i])
            # inside car
            if self.check_intersect(5, pts) and not ignore_in:
                #print("Enter Area")
                if (id[i] not in self.data_en_c) and (cls[i] == 2):
                    if (id[i] in self.data_whe):
                        self.count[7] += 1 
                        if id[i] not in self.data_car_count:
                            self.count[2]+=1 
                            self.data_car_count.append(id[i])
                    self.data_en_c.append(id[i])
            # outside
            # cam_error -> to avoid cctv jumping frame problem cuz the extra count
            # data_ou -> if person occur add into list (depend on person conf) -> add enter person
            # data_ou_count -> if person occur add into list (depend on out conf) -> add person
            # data_car_count -> if car occur add into list (depend on car conf) -> add car
            # data_whe -> list to store vehicle
            #          -> add directly into list (depend on out conf) -> add enter car
            if self.check_intersect(4, pts) and not ignore_out:
                if cls[i] == 0:
                    if (id[i] not in self.data_ou) :
                        self.data_ou.append(id[i])
                    if id[i] not in self.data_ou_count and (conf[i] > self.out_conf - 0.01) and (id[i] not in self.data_en_p) and not cam_error:
                        self.count[1]+=1
                        self.data_ou_count.append(id[i]) 
                elif cls[i] == 2:
                    if (id[i] not in self.data_whe) :
                        self.data_whe.append(id[i])
                    if id[i] not in self.data_car_count and (conf[i] > self.out_conf - 0.01) and (id[i] not in self.data_en_c) and not cam_error:
                        self.count[2]+=1
                        self.data_car_count.append(id[i]) 
                else:
                    if (id[i] not in self.data_whe):
                        self.data_whe.append(id[i])
                        if not cam_error:
                            self.count_cls(cls[i])

#        print("AFTER\n-----------------------------------")
#        print("Enter list: ", self.data_en_p)
#        print("Outside list: ", self.data_ou)
#        print("Wheel list: ", self.data_whe)
        
                
    
    def check_space(self):
        if len(self.data_en_p) >= 40 :
            del self.data_en_p[:28]
        if len(self.data_en_c) >= 40 :
            del self.data_en_c[:28]
        if len(self.data_whe) >= 50:
            del self.data_whe[:35]
        if len(self.data_ou) >= 40 :
            del self.data_ou[:28]
        if len(self.data_ou_count) >= 40 :
            del self.data_ou_count[:28]
        if len(self.data_car_count) >= 40 :
            del self.data_car_count[:28]
    
    def load_boxes(self):
        # 0 -> inside_P
        # 1 -> outside
        # 2 -> ignore_IN
        # 3 -> ignore_OUT
        # 4 -> ignore_IO
        # 5 -> inside_C
        # get the current working directory
#        print(Path(str(self.args.project)[:-10] + "source\out.txt"))
        self.xlx_path = self.createFile(str(Path.cwd()) +'\\')
        text = str(Path.cwd()) + "\\out.txt"
        with open(text) as f:
            lines = f.readlines()
            for line in lines:
                word = line.strip().split(' ')
                if word[0] =='d':
                    if len(word) > 1:
                        self.date = word[1]
                    continue
                if word[0] =='t':
                    self.time = datetime.strptime(word[1], '%H:%M:%S')
                    continue
                if word[0] =='p':
                    self.person_conf = float(word[1])
                    continue
                if word[0] =='c':
                    self.car_conf = float(word[1])
                    continue
                if word[0] =='o':
                    self.out_conf = float(word[1])
                    continue
                if word[0] =='f':
                    self.vid_stride = int(word[1])
                    continue
                if word[0] =='r':
                    self.display_ratio = int(word[1])
                    continue
#                print(f"p: {self.person_conf}\nc: {self.car_conf}")
                xy = [eval(word[1]), eval(word[2])]
                if word[0] == '0':
                    self.inside_P.append(xy)
                if word[0] == '1':
                    self.outside.append(xy)
                if word[0] == '2':
                    self.ignore_IN.append(xy)
                if word[0] == '3':
                    self.ignore_OUT.append(xy)
                if word[0] == '4':
                    self.ignore_IO.append(xy)
                if word[0] == '5':
                    self.time_region.append(xy)
                if word[0] == '6':
                    self.inside_C.append(xy)
        
    def draw_boxes(self, plotted_img):
        result = f"Person: {self.count[1]} | Enter Person: {self.count[0]} | Car: {self.count[2]} | Enter Car: {self.count[7]} | Motor: {self.count[3]} | Bus: {self.count[4]} | Truck: {self.count[5]} | Bic: {self.count[6]}"
        cv2.putText(plotted_img, result, org=(10,20), fontFace=0,
                     fontScale=0.5*self.display_ratio, thickness = 2*self.display_ratio, color=(255, 255, 255))
        if self.ignore_IO:
            for j in (self.ignore_IO):
                cv2.rectangle(plotted_img, j[0], j[1], color=(165,42,42), thickness=2, lineType=cv2.LINE_AA)
        if self.ignore_IN:
            for j in (self.ignore_IN):
                cv2.rectangle(plotted_img, j[0], j[1], color=(65, 105, 225), thickness=2, lineType=cv2.LINE_AA)
        if self.inside_C:
            for j in (self.inside_C):
                cv2.rectangle(plotted_img, j[0], j[1], color=(95,158,160), thickness=2, lineType=cv2.LINE_AA)
        if self.ignore_OUT:
            for j in (self.ignore_OUT):
                cv2.rectangle(plotted_img, j[0], j[1], color=(51, 68, 255), thickness=2, lineType=cv2.LINE_AA)
        if self.inside_P:
            for j in (self.inside_P):
                cv2.rectangle(plotted_img, j[0], j[1], color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        if self.outside:
            for j in (self.outside):
                cv2.rectangle(plotted_img, j[0], j[1], color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)


        
    def check_Frame(self, frame, dataset_frames):
        frames = dataset_frames
        #hour = 3600 * self.dataset.fps
        #hour = frames//24
        interval = 0
#        if (frame % 4500 == 0):
        #if (frame == 1):
         #   interval = 0
        #elif (frame % hour == 0) :
          #  interval = 1
        #elif  (frame == frames):
           # interval = (frame % hour)/hour
        #else:
            #return
        if frame == frames:
            self.detectTime(interval)
    
    def createFile(self, root, name = "report", i = 0, ex = ".xlsx"):

        if i == 1:
            name = name + f"({i})"
        if os.path.exists(root + name + ex):
            if i > 1:
                name = name[:-3] + f"({i})"
            return self.createFile(root, name, i+1)
        return root + name + ex
    
    def extract_time_rescale(self, img, scale_percent=50, custom_config = r'-c tessedit_char_whitelist=1234567890: --psm 6'):
        '''take path of image and return the extracted text on top right'''
        height, width, _ = img.shape
        box = self.time_region
        img_time = img[int(box[0][1]) : int(box[1][1]), int(box[0][0]) : int(box[1][0])] #only time hour:min:sec
        # grayscale
        img_time = cv2.cvtColor(img_time, cv2.COLOR_BGR2GRAY)
        # thresholding (>127 = black, else white)
        _, img_time = cv2.threshold(img_time, 240, 255, cv2.THRESH_BINARY_INV )
        # invert white and black
        img_time = cv2.bitwise_not(img_time)
        str_time = pytesseract.image_to_string(img_time, config=custom_config)
        print(str_time) 
        return
    
    def detectTime(self, interval):
        #print(self.df)
        count = self.count - self.prev_count
        #print("prev: ", self.prev_count)
        #print("current: ", self.count)
        #print("substract:", count)
#        self.time += timedelta(hours=1)
        self.time += timedelta(hours=interval)
        
        
        # save data
        data = {"Timestamp":str(self.time.time()),
                "Person": count[1], 
                "Enter Person":count[0],
                "Car":count[2],
                "Enter Car":count[7],
                "Motorcycle":count[3],
                "Bus":count[4],
                "Truck":count[5],
                "Bicycle":count[6]}
        self.df = self.df._append(data, ignore_index=True)
        self.prev_count = self.count.copy()
        
#        total_row=pd.DataFrame(self.df.sum(),columns=['Total']).T
#        self.df = pd.concat([self.df,total_row])
        self.df.to_excel(self.xlx_path)

        #print(self.df)
        #count = self.count - self.prev_count
        #print("prev: ", self.prev_count)
        #print("current: ", self.count)
        #print("substract:", count)





class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = self.get_save_dir()
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self.mode = None

        callbacks.add_integration_callbacks(self)
      

    def get_save_dir(self):
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        return increment_path(Path(project) / name, exist_ok=self.args.exist_ok)

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def inference(self, im, *args, **kwargs):
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(im, augment=self.args.augment, visualize=visualize)

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x) for x in im]


    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass


    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.mode.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    
    def on_predict_postprocess_end(self):
        """Postprocess detected boxes and update with object tracking."""
        bs = self.dataset.bs
        im0s = self.batch[1]
        for i in range(bs):
            det = self.results[i].boxes.data.cpu().numpy()
            #print(type(det))
            if len(det) == 0:
                continue
            tracks = self.trackers[i].update(det, im0s[i])
            if len(tracks) == 0:
                continue
            idx = tracks[:, -1].astype(int)
            self.results[i] = self.results[i][idx]
            self.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        
        self.mode = ObjectDetection()
        
        self.mode.load_boxes()
#        print("inside_P: ", self.inside_P)
#        print("outside: ", self.outside)
#        print("ignore in: ", self.ignore_IN)
#        print("ignore out: ", self.ignore_OUT)
#        print("igonore io: ", self.ignore_IO)

        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)
            

        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')

        # ImageLoader as iterator
        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im0s, vid_cap, s = batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.inference(im, *args, **kwargs)

            # Postprocess
            with profilers[2]:
                if isinstance(self.model, AutoBackend):
                    self.results = self.postprocess(preds, im, im0s)
                else:
                    self.results = self.model.postprocess(path, preds, im, im0s)
            with profilers[3]:
                self.run_callbacks('on_predict_postprocess_end')
                
            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.seen += 1
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n,
                    'tracking': profilers[3].dt * 1E3 / n}
                p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                p = Path(p)
#                print(self.results[i].speed)
                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))
                if self.args.save or self.args.save_txt:
                    self.results[i].save_dir = self.save_dir.__str__()
                if self.args.show and self.plotted_img is not None:
                    self.show(p)
                if self.args.save and self.plotted_img is not None:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            
            

            self.run_callbacks('on_predict_batch_end')
            yield from self.results


            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape '
                        f'{(1, 3, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#        w, h = im0.shape[1], im0.shape[0]
        self.mode.draw_boxes(im0)
        (w,h) = (int(self.dataset.w/self.mode.display_ratio), int(self.dataset.h/self.mode.display_ratio))
        cv2.imshow(str(p), cv2.resize(im0, (w,h)))
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        self.mode.draw_boxes(im0)
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = self.dataset.fps  # integer required, floats produce error in MP4 codec
                    w = self.dataset.w
                    h = self.dataset.h
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = '.mp4' if MACOS else '.avi' if WINDOWS else '.avi'
                fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'MJPG'
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)
        

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()
            

        """"
        print(self.time)
        print(self.df)
        
        print("data_list length: ", len(self.data_list))
        if len(self.data_list) > 5:
            import time
            print(self.data_list)
            del(self.data_list[:3])
            print(self.data_list)
            self.data_list.append(999999)
            print(self.data_list)
            time.sleep(10)
        """
        if result.boxes.id is not None:
            conf = result.boxes.conf
            id = result.boxes.id.type(torch.int32)
            cls = result.boxes.cls.type(torch.int32)
            boxes = result.boxes.xywh.type(torch.int32)
            centre = []
            valid_id = []
            valid_cls = []
            valid_conf = []
            for i, j in enumerate(boxes):
                if self.mode.check_conf(cls[i], conf[i]) or self.dataset.error:
                    centre.append((boxes[i][0], boxes[i][1]) )
                    valid_id.append(id[i])
                    valid_cls.append(cls[i])
                    valid_conf.append(conf[i])
                    
 #           print(self.df)
 #           print("id: ", id)
 #           print("cls: ", cls)
 #           print("xywh", boxes)
 #           print("centre:", centre)
 #           print("is tracked: ", result.boxes.is_track)
 #           print(self.data_list)
 #           result.boxes.is_count = True
 #           print()
 #           print(self.count)
 #           print("conf: ", conf)
#            print(f"VALID ID: {valid_id}\nVALID CLASS: {valid_cls}\n--------------------------")
            self.mode.count_obj(centre, valid_cls, valid_id, valid_conf, self.dataset.error)
            
        self.mode.check_Frame(frame, self.dataset.frames)
            

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops',
                             file_name=self.data_path.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))

        return log_string

