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

from ultralytics.cfg import get_cfg
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, SETTINGS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path

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

        callbacks.add_integration_callbacks(self)
        # 0 enter_person
        # 1 person
        # 2 car
        # 3 motorcycle
        # 4 bus
        # 5 truck
        # 6 bicycle
        self.count = np.zeros(2, dtype = int)
        self.prev_count = np.zeros(2, dtype = int)
#        self.data_list= []
#        self.data_en= []
#        self.data_ou = []
        self.cashier = None 
        self.customer = None
        self.time = None
        self.cash = None
        self.customer_pay = None
        self.df = pd.DataFrame(columns = ['Timestamp', 'Cash Payment', 'Customer'])

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

    def downscaleResize(self, img, scale_percent):
        ''' input: image object, scale_percent or ratio
            output: image object with resized ratio'''
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)
        return resized

    def extract_time_rescale(self,  img, scale_percent=50, custom_config = r'-c tessedit_char_whitelist=1234567890Oo --psm 6'):
        '''take path of image and return the extracted text on top right'''

        img_time = img[20:55, 1122:1258] #only time hour:min:sec
        img_time = self.downscaleResize(img_time, scale_percent)
        img_time = cv2.cvtColor(img_time, cv2.COLOR_BGR2GRAY)   
        str_time = pytesseract.image_to_string(img_time, config=custom_config)

        return str(str_time)

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

        """
        if (self.time is None or frame % 54000 == 0 ):
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            im = result.plot(**plot_args)
            self.detectTime(im)
        """
        
        """
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
            id = result.boxes.id.type(torch.int64)
            cls = result.boxes.cls.type(torch.int64)
            boxes = result.boxes.xywh.type(torch.int64)
            centre = []
            boxes_area = result.boxes.xyxy.type(torch.int64)
            for i, j in enumerate(boxes):
                if cls[i] == 0:
                    boxes[i][1] = (boxes[i][1] + boxes_area[i][3])//2
                centre.append((boxes[i][0], boxes[i][1]) )
#            print("id: ", id)
#            print("cls: ", cls)
#            print("xywh", boxes)
#            print("xyxy: ", boxes_area)
#            print("centre:", centre)
#            print("is tracked: ", result.boxes.is_track)
#            print(self.data_list)
            result.boxes.is_count = True
            print()
            print(self.count)
            if (self.count_obj(centre, cls, id, boxes_area)):
                plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
                if not self.args.retina_masks:
                    plot_args['im_gpu'] = im[idx]
                im = result.plot(**plot_args)
                self.detectTime(im)
        if frame == self.dataset.frames:
            plot_args = {
            'line_width': self.args.line_width,
            'boxes': self.args.boxes,
            'conf': self.args.show_conf,
            'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            im = result.plot(**plot_args)
            self.detectTime(im)
            xlsx = str(Path.cwd()) + "\\out.xlsx"
            total_row=pd.DataFrame(self.df.sum(),columns=['Total']).T
            self.df = pd.concat([self.df,total_row])
            self.df.to_excel(xlsx)
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

    def compare_index(self, p1, p2):
        x_point = (p1[0], p2[0]) if p1[0] <p2[0] else (p1[0], p2[0])
        y_point = (p1[1], p2[1]) if p1[1] <p2[1] else (p1[1], p2[1])
        return (x_point, y_point)
    
    def count_cls(self, cls,id):
        # person
        if cls == 0:
            self.count[1]+=1
            self.data_ou.append(id)
        # bicycle
        elif cls == 1:
            self.count[6]+=1
        # car
        elif cls == 2:
            self.count[2]+=1
        # motorcycle
        elif cls == 3:
            self.count[3]+=1
        # bus
        elif cls == 5:
            self.count[4]+=1
        # truck
        elif cls == 7:
            self.count[5]+=1
    
    
    def check_space(self):
        if len(self.data_en) >= 100 :
            del self.data_en[:80]
        if len(self.data_list) >= 150:
            del self.data_list[:120]
        if len(self.data_ou) >= 100 :
            del self.data_ou[:80]

    def count_obj(self, centre, cls, id, boxes_area):
        cashier = self.cashier
        customer = self.customer
        cashier_point = self.compare_index(cashier[0], cashier[1])
        customer_point = self.compare_index(customer[0], customer[1])
        isPersonCashier = False
        isPersonCustomer = False
#        self.check_space()
        print("cash:", self.cash)
        for i, j in enumerate(centre):
            # enter person counting
            if self.customer_pay:
                if id[i] == self.customer_pay:
                    if j[0] in range(customer_point[0][0], customer_point[0][1]) and j[1] in range(customer_point[1][0], customer_point[1][1]):
                        pass
                    else:
                        self.customer_pay = None
                        print("customer out")
                    #if (boxes_area[i][0] > customer_point[0][1]) or (boxes_area[i][2] < customer_point[0][0]) or (boxes_area[i][1] > customer_point[1][0]) or (boxes_area[i][3] < customer_point[1][1]):

            else:
                if j[0] in range(customer_point[0][0], customer_point[0][1]) and j[1] in range(customer_point[1][0], customer_point[1][1]):
                    print("hello worlding")
                    if cls[i] == 1 and self.cash is None:
                        print("hello world")
                        print(j)
                        self.cash = j
                    if cls[i] == 0 and self.cash is not None:
                        if self.cash[0] in range(boxes_area[i][0], boxes_area[i][2]) and self.cash[1] in range(boxes_area[i][1], boxes_area[i][3]):
                            self.customer_pay = id[i]
                            isPersonCustomer = True
                        #if (id[i] not in self.data_list) and (id[i] not in self.data_en):
                        #    self.data_list.append(id[i])
                        #    self.count_cls(cls[i], id[i])
            if j[0] in range(cashier_point[0][0], cashier_point[0][1]) and j[1] in range(cashier_point[1][0], cashier_point[1][1]):
                    if cls[i] == 0:
                        isPersonCashier = True
        
        if(isPersonCashier and isPersonCustomer):    
            print("Payment happening, sending data")
#                current = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                time_database.append(current)
            isPersonCustomer = False
            self.cash = None
            self.count[0] += 1
            import time
            time.sleep(5)
            return True
        return False
            
                        

    
    def detectTime(self, im):
        #print(self.df)
        count = self.count
        #print("prev: ", self.prev_count)
        #print("current: ", self.count)
        #print("substract:", count)
        time = self.extract_time_rescale(im)
        self.time = f"{time[0:2]}:{time[2:4]}:{time[4:6]}"
        
        self.save_data(count)
        #print(self.df)
        #count = self.count - self.prev_count
        #print("prev: ", self.prev_count)
        #print("current: ", self.count)
        #print("substract:", count)

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

    def save_data(self, count):
        data = {"Timestamp":self.time,
                "Cash Payment": count[0], 
                "Customer":count[1]}
        self.df = self.df._append(data, ignore_index=True)
        #self.prev_count = self.count.copy()

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        
        # get the current working directory
        text = str(Path.cwd()) + "\\out.txt"
        with open(text) as f:
            lines = f.readlines()
            for line in lines:
                word = line.strip().split(' ')
                xy = [eval(word[1]), eval(word[2])]
                if word[0] == '0':
                    self.cashier = xy
                else:
                    self.customer = xy
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
                    'tracking': profilers[2].dt * 1E3 / n}
                p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                p = Path(p)

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
        w, h = im0.shape[1], im0.shape[0]
        customer = self.customer
        #result = f"Person: {self.count[1]} | Enter Person: {self.count[0]} | Car: {self.count[2]} | Motor: {self.count[3]} | Bus: {self.count[4]} | Truck: {self.count[5]} | Bic: {self.count[6]}"
        result = f"Cash: {self.count[0]} | Customer: {self.count[1]}"
        cv2.rectangle(im0, self.cashier[0], self.cashier[1], color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(im0, customer[0], customer[1], color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(im0, result, org=(10,20), fontFace=0,
                     fontScale=0.5, color=(255, 255, 255))
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        customer = self.customer
        result = f"Cash: {self.count[0]} | Customer: {self.count[1]}"
        cv2.line(im0, self.cashier[1], self.cashier[1], color=(0, 255, 0), thickness = 2)
        cv2.rectangle(im0, self.cashier[0], self.cashier[1], color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(im0, customer[0], customer[1], color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(im0, result, org=(10,20), fontFace=0,
                     fontScale=0.5, color=(255, 255, 255))
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    
    """Check cash transaction"""
cash_transaction = []
time_database = []
cashier_id = -1
customer_id = -1
is_transaction_sent = False
print("Cashierid: ", cashier_id)
def check_intersect(img, roi, obj_box, id, cat, names, box_info):
    global cashier_id
    global customer_id
    global is_transaction_sent
    print("transaction happening: ", is_transaction_sent)
    isPersonCashier = False
    isPersonCustomer = False
    isCash = False
    isSameId = False

    print("box info")
    print(box_info)
    # Compare in specific frame.
    for i, (c, x1, y1, x2, y2) in enumerate(roi):
                # x1, y1, x2, y2, category, identity
        for j, (xa, ya, xb, yb, cate, iden)  in enumerate(box_info):

            # print(f"i is {i}, j is {j} and o is {o}")
            if((x1 > xb or x2 < xa or ya > y2 or yb < y1)):        # If boxes are not instersect with one another
                """
                isPersonCustomer to get the specific frame right after the customer leaves the customer box.
                int(c) == 1 to ensure the taken value is the customer box
                int(iden) == int(customer_id) to get the specific customer that leaves the customer box
                """
                # Customer leaving
                if isPersonCustomer and int(c) == 1 and int(iden) == int(customer_id):  
                    print("Customer leaving")  
                    cv2.waitKey(2000)
                    isPersonCustomer = False
                    customer_id = -1
                    is_transaction_sent = False
                # Cashier leaving
                if isPersonCashier and int(c) == 0 and int(iden) == int(cashier_id):   
                    print("Cashier leaving")
                    customer_id = -1
                    isPersonCashier = False  

            else:       
                """
                cate = Person     c:0 = Cashier 
                cate = Cash       c:1 = Customer
                """                # At least one intersection
                if(int(cate) == 0 and int(c) == 0):       # Person in cashier area
                    # print("Got cashier")
                    if cashier_id == -1:
                        cashier_id = iden
                    isPersonCashier = True
                elif(int(cate) == 0 and int(c) == 1):     # Person in customer area
                    # print("Got customer")
                    if customer_id == -1:
                        customer_id = iden
                    isPersonCustomer = True
                elif(int(cate) == 1 and int(c)==1):      # if the detection category is cash (1) and the box is customer, indicate customer start paying
                    # print("Got cash")
                    # print("id[j]: ", id[j])
                    cash_id[j] = id[j]
                    cash_temp.append(id[j])
                    isCash = True
                else:           # for matching of cashier and customer with the boxes
                    print(f"{names[cat[j]]}th object is in {c}th box")
    print("cashier id: ", cashier_id)
    print("customer id: ", customer_id)
    # print(isPersonCashier)
    # print(isPersonCustomer)
    # print(isCash)
    # print("cashtemp is: ", cash_temp)
    if(isPersonCashier and isPersonCustomer and isCash):

        
        if not is_transaction_sent:
            print("Payment happening, sending data")

            current = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_database.append(current)

            cv2.waitKey(1000)  
            is_transaction_sent = True     
        