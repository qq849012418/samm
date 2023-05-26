"""
MIT License
Copyright (c) 2023 Yihao Liu
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml, cv2, os, pickle, zmq, json, shutil
from datetime import datetime
import traceback
import logging
import torch
import SimpleITK as sitk
from skimage import transform

class sam_server():

    def __init__(self):

        # Latency logging
        # log latency?
        self.flag_loglat = False
        if self.flag_loglat:
            now = datetime.now()
            self.logctrmax = 300
            self.timearr_RCV_INF = [now for idx in range(self.logctrmax)]
            self.timearr_CPL_INF = [now for idx in range(self.logctrmax)]
            self.timearr_EMB = [now, now]

        # create a workspace
        workspace = os.path.dirname(os.path.abspath(__file__))
        workspace = os.path.join(workspace, 'samm-workspace')
        if not os.path.exists(workspace):
            os.makedirs(workspace)
        self.workspace = workspace

        # check if model exists
        # self.sam_checkpoint = self.workspace + "/sam_vit_h_4b8939.pth"
        self.sam_checkpoint = self.workspace + "/medsam_20230423_vit_b_0.0.1.pth"

        if not os.path.isfile(self.sam_checkpoint):
            raise Exception("SAM model file is not in " + self.sam_checkpoint)
        
        # Load the segmentation model
        # self.model_type = "vit_h"
        self.model_type = "vit_b"
        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA detected.")
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("MPS detected.")
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

        # temp files, just for initialization, will be overwritten
        self.imgsize_path = self.workspace + "/imgsize"
        if not os.path.isfile(self.imgsize_path):
            f = open(self.workspace + "/imgsize", "w+")
            f.write("IMAGE_WIDTH: " + str(240) + "\n" + "IMAGE_HEIGHT: " + str(352) + "\n" )
            f.close()
        if not os.path.isfile(self.workspace + "/imgsize_input_size"):
            f = open(self.workspace + "/imgsize_input_size", "w")
            f.write("INPUT_WIDTH: " + str(698) + "\n" \
                + "INPUT_HEIGHT: " + str(1024) + "\n" )
            f.close()
        if not os.path.isfile(self.workspace + "/imgsize_original_size"):
            f = open(self.workspace + "/imgsize_original_size", "w")
            f.write("ORIGINAL_WIDTH: " + str(240) + "\n" \
                + "ORIGINAL_HEIGHT: " + str(352) + "\n" )
            f.close()
        if not os.path.isfile(self.workspace + "/config.yaml"):
            with open(self.workspace + "/config.yaml", 'w') as fp:
                pass

        # initialize some parameters for testing (assumes the embeddings are saved)
        self.predictor.is_image_set = True
        with open(os.path.join(self.workspace,"imgsize_input_size"), 'r') as file:
            yaml_file = yaml.safe_load(file)
        self.predictor.input_size = \
            (int(yaml_file["INPUT_WIDTH"]), int(yaml_file["INPUT_HEIGHT"]))
        with open(os.path.join(self.workspace,"imgsize_original_size"), 'r') as file:
            yaml_file = yaml.safe_load(file)
        self.predictor.original_size = \
            (int(yaml_file["ORIGINAL_WIDTH"]), int(yaml_file["ORIGINAL_HEIGHT"]))
        
        masks = np.full(self.predictor.original_size, False)
        memmap = np.memmap(os.path.join(self.workspace, 'mask.memmap'), dtype='bool', mode='w+', shape=masks.shape)
        memmap[:] = masks[:]
        memmap.flush()

        # create a folder to store slices
        self.slices_folder_path = os.path.join(self.workspace, 'slices')
        if not os.path.exists(self.slices_folder_path):
            os.makedirs(self.slices_folder_path)

    def cleanup(self):
        self.sock_rcv.close()

    def computeEmbedding(self):

        # create a folder to store segmented data
        output_folder = os.path.join(self.workspace, 'segmented_images')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        # read size of images
        with open(self.imgsize_path, 'r') as file:
            yaml_file = yaml.safe_load(file)
        image_width = int(yaml_file["IMAGE_WIDTH"])
        image_height = int(yaml_file["IMAGE_HEIGHT"])

        # Loop through all files in the folder
        for filename in os.listdir(self.slices_folder_path):

            data = np.memmap(os.path.join(self.slices_folder_path, filename), \
                dtype='float64', mode='r+') 
            data = data.reshape((image_width,image_height,1))# reshape
            data = 255 * data / data.max()
            data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.predictor.set_image(data)
            
            # store all pred_data according to the filename in a folder
            pkl_file = os.path.join(output_folder, "segmented_" + str(filename) + ".pkl")
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.predictor.features, f)
                print(f'Predictor successfully saved to "{f}"')

        f = open(os.path.join(self.workspace, "imgsize_input_size"), "w")
        f.write("INPUT_WIDTH: " + str(self.predictor.input_size[0]) + "\n" \
            + "INPUT_HEIGHT: " + str(self.predictor.input_size[1]) + "\n" )
        f.close()
        f = open(os.path.join(self.workspace, "imgsize_original_size"), "w")
        f.write("ORIGINAL_WIDTH: " + str(self.predictor.original_size[0]) + "\n" \
            + "ORIGINAL_HEIGHT: " + str(self.predictor.original_size[1]) + "\n" )
        f.close()

        print("All images have been processed.")

    def load_feature(self, feature_path: str):
        self.feature_path = feature_path
        with open(self.feature_path, 'rb') as f:
            features = pickle.load(f)
        
        self.predictor.features = features
    
    def predict(self, input_point:np.ndarray, input_label:np.ndarray, box:np.ndarray, input_mask:np.ndarray):
        self.input_point = input_point
        self.input_label = input_label
        self.masks, self.scores, self.logits = \
            self.predictor.predict( \
                point_coords=input_point, \
                point_labels=input_label, \
                box=box, \
                mask_input=input_mask, \
                multimask_output=True )

    def infer_image(self, input_point, input_label, input_mask, image_name):
        # input_point = np.array([[200, 100]])
        # input_label = np.array([1])
        slice_idx = "".join(list(filter(str.isdigit, image_name)))
        slice_idx = int(slice_idx)
        if len(input_label) != 0:
            self.load_feature(os.path.join(self.workspace, "segmented_images", "segmented_" + image_name + ".pkl"))
            ## original:only point
            # self.predict(input_point,input_label)

            ##  Keenster1:msk
            # msk_slice_i = input_mask[slice_idx, :, :]
            # msk_slice_i = transform.resize(msk_slice_i, (256, 256), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
            # self.predict(None, None, msk_slice_i[None,:,:])

            ## Keenster2:box
            msk_slice_i = input_mask[slice_idx, :, :]
            x, y, w, h = cv2.boundingRect(msk_slice_i.astype(np.uint8))
            bbox=np.array([x, y, x+w, y+h])
            self.predict(None, None, bbox, None)

            ## Keenster3:box+mask
            # msk_slice_i = input_mask[slice_idx, :, :]
            # x, y, w, h = cv2.boundingRect(msk_slice_i.astype(np.uint8))
            # bbox=np.array([x, y, x+w, y+h])
            # msk_slice_i = transform.resize(msk_slice_i, (256, 256), order=0, preserve_range=True, mode='constant',
            #                                anti_aliasing=True)
            # self.predict(None, None, bbox, msk_slice_i[None,:,:])
        else:
            self.masks = np.full(self.predictor.original_size, False)
        # self.imageshow(self.workspace + "/slices/" + image_name)

        #Keenster:mask postprocessing
        outmask=self.masks[0][:].astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)  # 设置kenenel大小
        erosion = cv2.erode(outmask, kernel, iterations=3)  # 腐蚀去除白噪点
        cv2.imshow("erosion", erosion)
        dilate = cv2.dilate(erosion, kernel, iterations=3)  # 膨胀还原图形
        x, y, w, h = cv2.boundingRect(dilate)
        print(image_name + " x:" + str(x) + " y:" + str(y) + " w:" + str(w) + " h:" + str(h))
        if w*h<10 or w*h>=512*512/4:
            print(image_name+" SAM output too small or too large,use matlab prior as final output")
            dilate=input_mask[slice_idx, :, :].astype(np.uint8)
        memmap = np.memmap(os.path.join(self.workspace, "mask"+str(slice_idx)+".memmap"), dtype='bool', mode='w+', shape=self.masks[0].shape)
        # memmap[:] = self.masks[0][:]
        memmap[:] = dilate
        memmap.flush()
        return dilate

    def imageshow(self, image_path):
        with open(self.imgsize_path, 'r') as file:
            yaml_file = yaml.safe_load(file)
        image_width = int(yaml_file["IMAGE_WIDTH"])
        image_height = int(yaml_file["IMAGE_HEIGHT"])
        data = np.fromfile(image_path,dtype=np.float64)
        data = data.reshape((image_width,image_height,1))# reshape
        image = 255 * data / data.max()
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca())
            self.show_points(self.input_point,self.input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show() 

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def main():

    #Keenster: load mask from matlab
    print("Loading mask from SSM  ... ")
    msk_sitk = sitk.ReadImage("D:\\Keenster\\MatlabScripts\\KeensterSSM\\XH_01\\XH_01_erector_spinae_right_SSMpredictfinal.nii.gz")
    msk_data = sitk.GetArrayFromImage(msk_sitk)

    flag_loop = True
    first_flag=True
    print("Initializing SAM server  ... ")
    srv = sam_server()
    print("SAM server initialized ... ")
    context = zmq.Context()
    zmqsocket = context.socket(zmq.PULL)
    zmqsocket.bind("tcp://*:5555")
    zmqsocket.setsockopt(zmq.RCVTIMEO, 30)
    srv.sock_rcv = zmqsocket

    print("Starting To Wait for Messages ... ")

    # Time log
    if srv.flag_loglat:
        ctr_RCV_INF = 0
        ctr_CPL_INF = 0

    while flag_loop:
        
        # Main loop
        try:
            # Recv msg
            msg = json.loads(srv.sock_rcv.recv_json())
            # Embedding command
            if msg["command"] == "COMPUTE_EMBEDDING":
                if srv.flag_loglat:
                    srv.timearr_EMB[0] = datetime.now()
                srv.computeEmbedding()

                ## Keenster 一次性全部识别
                if first_flag:
                    first_flag = False
                    for i in range(np.size(msk_data,axis=0)):
                        msk_data[i,:,:]=srv.infer_image( \
                            np.array([666]), \
                            np.array([666]),\
                            msk_data, \
                            "slc" + str(i))
                    # 输出
                    final_sitk=sitk.GetImageFromArray(msk_data)
                    final_sitk.CopyInformation(msk_sitk)
                    # 将修改后的图像保存到磁盘
                    sitk.WriteImage(final_sitk, "D:\\Keenster\\MatlabScripts\\KeensterSSM\\XH_01\\XH_01_erector_spinae_right_SSMSAM"+datetime.now().strftime('%m%d%H%M')+".nii.gz")

                if srv.flag_loglat:
                    srv.timearr_EMB[1] = datetime.now()
                    file_name = srv.workspace + "timearr_EMB.pkl"
                    with open(file_name, 'wb') as file:
                        pickle.dump(srv.timearr_EMB, file)
                        print("Time for embedding computing is saved.")
            # Inference command
            if msg["command"] == "INFER_IMAGE":
                if srv.flag_loglat:
                    srv.timearr_RCV_INF[ctr_RCV_INF] = datetime.now()
                    ctr_RCV_INF = ctr_RCV_INF + 1
                srv.infer_image( \
                    np.array(msg["parameters"]["point"]), \
                    np.array(msg["parameters"]["label"]), \
                    msk_data,\
                    msg["parameters"]["name"])

                if srv.flag_loglat:
                    srv.timearr_CPL_INF[ctr_CPL_INF] = datetime.now()
                    ctr_CPL_INF = ctr_CPL_INF + 1
                    if ctr_RCV_INF >= srv.logctrmax - 1 or ctr_CPL_INF >= srv.logctrmax - 1:
                        file_name = srv.workspace + "/timearr_RCV_INF.pkl"
                        with open(file_name, 'wb') as file:
                            pickle.dump(srv.timearr_RCV_INF, file)
                        file_name = srv.workspace + "/timearr_CPL_INF.pkl"
                        with open(file_name, 'wb') as file:
                            pickle.dump(srv.timearr_CPL_INF, file)
                        print("Time for inference is saved.")
                        break
        except zmq.error.Again:
            continue
        except KeyboardInterrupt:
            flag_loop = False
            srv.cleanup()
        except Exception as e:
            logging.error(traceback.format_exc())
        
if __name__=="__main__":
    main()