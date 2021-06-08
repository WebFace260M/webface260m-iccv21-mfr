import argparse
import cv2
import time
import json
import sys
import numpy as np
import pickle
import os
import glob
from numpy.linalg import norm as l2norm
import datetime
from skimage import transform as trans
import face_detection
import os.path as osp
import onnxruntime
import onnx
from onnx import numpy_helper
from decimal import Decimal, ROUND_HALF_UP


arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_norm(lmk, image_size = 112, mode='arcface'):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    #lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    tform.estimate(lmk, arcface_src)
    M = tform.params[0:2,:]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    return warped

class PyWebFace260M:

    def __init__(self, use_gpu=False):
        self.image_size = (112,112)
        self.det_size = 224
        self.do_flip = False
        self.iter = 0
        self.use_gpu = use_gpu
        self.nodet = 0

        print("use_gpu: {}".format(use_gpu))

    def load(self, rdir):
        det_model = os.path.join(rdir, 'det', 'R50-retinaface.onnx')
        self.detector = face_detection.get_retinaface(det_model)
        print('use onnx-model:', det_model)

        self.detector.prepare(use_gpu=self.use_gpu)
        max_time_cost = 1000

        self.model_file = os.path.join(rdir, 'face_reg', "R18.onnx")
        print('use onnx-model:', self.model_file)



        if self.use_gpu:
            session = onnxruntime.InferenceSession(self.model_file)
        else:
            sessionOptions = onnxruntime.SessionOptions()
            sessionOptions.intra_op_num_threads = 1
            sessionOptions.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(self.model_file, sess_options=sessionOptions)
            print("face reg intra_op_num_threads {} inter_op_num_threads {}".format(sessionOptions.intra_op_num_threads, sessionOptions.inter_op_num_threads))


        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        print('input-shape:', input_shape)
        if len(input_shape) != 4:
            return "length of input_shape should be 4"
        if not isinstance(input_shape[0], str):
            # return "input_shape[0] should be str to support batch-inference"
            print('reset input-shape[0] to None')
            model = onnx.load(self.model_file)
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            new_model_file = osp.join(self.model_path, 'zzzzrefined.onnx')
            onnx.save(model, new_model_file)
            self.model_file = new_model_file
            print('use new onnx-model:', self.model_file)
            try:
                session = onnxruntime.InferenceSession(self.model_file, None)
            except:
                return "load onnx failed"

            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
            print('new-input-shape:', input_shape)

        self.image_size = tuple(input_shape[2:4][::-1])
        # print('image_size:', self.image_size)
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
            # print(o.name, o.shape)
        if len(output_names) != 1:
            return "number of output nodes should be 1"
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        # print(self.output_names)
        model = onnx.load(self.model_file)
        graph = model.graph
        if len(graph.node) < 8:
            return "too small onnx graph"

        input_size = (112, 112)
        self.crop = None
        if input_size != self.image_size:
            return "input-size is inconsistant with onnx model input, %s vs %s" % (input_size, self.image_size)


        input_mean = None
        input_std = None
        if input_mean is not None or input_std is not None:
            if input_mean is None or input_std is None:
                return "please set input_mean and input_std simultaneously"
        else:
            find_sub = False
            find_mul = False
            for nid, node in enumerate(graph.node[:8]):
                print(nid, node.name)
                # if node.name.startswith('Sub') or node.name.startswith('_minus'):
                #     find_sub = True
                # if node.name.startswith('Mul') or node.name.startswith('_mul'):
                #     find_mul = True
                if "_sub" in node.name or "_minus" in node.name:
                    find_sub = True
                if "_mul" in node.name:
                    find_mul = True

            print("find_sub {} find_mul {}".format(find_sub, find_mul))
            if find_sub and find_mul:
                # mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                input_mean = 127.5
                input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        for initn in graph.initializer:
            weight_array = numpy_helper.to_array(initn)
            #             if weight_array.dtype!=np.float32:
            #                 return 'all weights should be float32 dtype'
            dt = weight_array.dtype
            if dt.itemsize < 4:
                return 'invalid weight type - (%s:%s)' % (initn.name, dt.name)
        test_img = None
        if test_img is None:
            test_img = np.random.randint(0, 255, size=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            test_img = cv2.resize(test_img, self.image_size)
        feat, cost, det_cost, face_reg_cost = self.benchmark(test_img)

        self.feat_dim = feat.shape[0]
        cost_ms = int(cost * 1000)
        det_cost_ms = int(det_cost * 1000)
        face_reg_cost_ms = cost_ms-det_cost_ms

        print("total time cost: %d ms" % cost_ms)
        print("det time cost: %d ms" % det_cost_ms)
        print("face_reg time cost: %d ms" % face_reg_cost_ms)

        if cost_ms > max_time_cost:
            print ("max time cost exceed, given %.4f" % cost_ms)
        self.cost_ms = cost_ms
        print('check stat:, feat-dim: %d, time-cost-ms: %.4f, input-mean: %.3f, input-std: %.3f' % (self.feat_dim, self.cost_ms, self.input_mean, self.input_std))
        return cost_ms, det_cost_ms, face_reg_cost_ms

    def benchmark(self, img):
        costs = []
        det_costs = []
        face_reg_costs = []

        for i in range(50):
            image_path = './{}.png'.format(i % 6)
            img = cv2.imread(image_path)
            ta = datetime.datetime.now()
            net_out, det_cost, face_reg_cost = self.get_feature_time(img)
            tb = datetime.datetime.now()
            cost = (tb-ta).total_seconds()
            costs.append(cost)
            det_costs.append(det_cost.total_seconds())
            face_reg_costs.append(face_reg_cost.total_seconds())

        index = np.argsort(costs)
        # costs = sorted(costs)
        print("costs: ", costs)
        cost = costs[index[5]]
        return net_out, cost, det_costs[index[5]], face_reg_costs[index[5]]

    def detect(self, img):
        S = self.det_size

        im_size_max = np.max(img.shape[0:2])
        det_scale = S * 1.0 / im_size_max
        str1 = str(det_scale * img.shape[0])
        str2 = str(det_scale * img.shape[1])

        height = int(Decimal(str1).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
        width = int(Decimal(str2).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        img_resize = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        # print("height", height)

        # if img.shape[1]>img.shape[0]:
        #     det_scale = float(S) / img.shape[1]
        #     width = S
        #     height = float(img.shape[0]) / img.shape[1] * S
        #     height = int(height)
        # else:
        #     det_scale = float(S) / img.shape[0]
        #     height = S
        #     width = float(img.shape[1]) / img.shape[0] * S
        #     width = int(width)
        # img_resize = cv2.resize(img, (width, height))
        img_det = np.zeros( (S,S,3), dtype=np.uint8)
        img_det[:height,:width,:] = img_resize
        bboxes, det_landmarks = self.detector.detect(img_det, threshold=0.3)
        bboxes /= det_scale
        det_landmarks /= det_scale
        return bboxes, det_landmarks

    def get_feature_time(self, img, im_type=0):
        face_reg_cost = 0
        ta = datetime.datetime.now()
        bboxes, det_landmarks = self.detect(img)

        def vis_img(img, points, faces, iter, color_j=(0, 0, 255), thickness=1, color_i=(0, 255, 0), radius=2):
            # box
            for face in faces:
                face_box = face  # ['data']
                face_box = [int(j) for j in face_box]

                cv2.rectangle(img, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color=color_j,
                              thickness=thickness)
            for kps in points:
                point = kps
                for j in point:
                    try:
                        p = j
                    except:
                        import pdb;
                        pdb.set_trace()
                    p = [int(jp) for jp in p]
                    cv2.circle(img, (p[0], p[1]), radius=radius, color=color_i, thickness=thickness)
            cv2.imwrite("./00_{}.jpg".format(iter), img)

        # vis_img(img, det_landmarks, bboxes, self.iter)
        # self.iter += 1

        tb = datetime.datetime.now()
        det_cost = tb - ta

        if bboxes.shape[0] == 0:
            return np.zeros((512,), dtype=np.float32), det_cost, ta-ta

        det = bboxes
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        box_cw = (det[:, 2] + det[:, 0]) / 2
        box_ch = (det[:, 3] + det[:, 1]) / 2
        dist_cw = box_cw - img.shape[1] / 2
        dist_ch = box_ch - img.shape[0] / 2
        score = area - (dist_cw ** 2 + dist_ch ** 2) * 2.0
        bindex = np.argmax(score)
        bbox = bboxes[bindex]
        det_landmark = det_landmarks[bindex]
        aimg = norm_crop(img, det_landmark)

        # print('cost det:', (tb - ta).total_seconds())
        # cv2.imwrite("./aimg_{}.jpg".format(self.iter), aimg)
        # ta = datetime.datetime.now()

        input_size = self.image_size
        if not isinstance(aimg, list):
            aimg = [aimg]
        blob = cv2.dnn.blobFromImages(aimg, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]

        feat = np.mean(net_out, axis=0)
        feat /= l2norm(feat)

        tc = datetime.datetime.now()
        # print('cost face reg:', (tb - ta).total_seconds())
        face_reg_cost = tc - tb
        return feat, det_cost, face_reg_cost

    def get_feature(self, img, im_type=0):
        # ta = datetime.datetime.now()
        bboxes, det_landmarks = self.detect(img)

        # print("bboxes {} det_landmarks {}".format(bboxes, det_landmarks))

        def vis_img(img, points, faces, iter, color_j=(0, 0, 255), thickness=1, color_i=(0, 255, 0), radius=2):
            # box
            for face in faces:
                face_box = face  # ['data']
                face_box = [int(j) for j in face_box]

                cv2.rectangle(img, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color=color_j,
                              thickness=thickness)
            for kps in points:
                point = kps
                for j in point:
                    try:
                        p = j
                    except:
                        import pdb;
                        pdb.set_trace()
                    p = [int(jp) for jp in p]
                    cv2.circle(img, (p[0], p[1]), radius=radius, color=color_i, thickness=thickness)
            cv2.imwrite("./00_{}.jpg".format(iter), img)

        # vis_img(img, det_landmarks, bboxes, self.iter)
        # self.iter += 1

        if bboxes.shape[0]==0:
            self.nodet += 1
            return np.zeros( (self.feat_dim,), dtype=np.float32 )
        det = bboxes
        area = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
        box_cw = (det[:,2]+det[:,0]) / 2
        box_ch = (det[:,3]+det[:,1]) / 2
        dist_cw = box_cw - img.shape[1]/2
        dist_ch = box_ch - img.shape[0]/2
        score = area - (dist_cw**2 + dist_ch**2)*2.0
        bindex = np.argmax(score)
        bbox = bboxes[bindex]
        det_landmark = det_landmarks[bindex]
        aimg = norm_crop(img, det_landmark)

        # tb = datetime.datetime.now()
        #
        # det_cost = tb - ta
        # print('cost det:', (tb - ta).total_seconds())
        # cv2.imwrite("./aimg_{}.jpg".format(self.iter), aimg)
        # ta = datetime.datetime.now()

        input_size = self.image_size
        if not isinstance(aimg, list):
            aimg = [aimg]
        blob = cv2.dnn.blobFromImages(aimg, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name : blob})[0]

        feat = np.mean(net_out, axis=0)
        feat /= l2norm(feat)

        # tc = datetime.datetime.now()
        # print('cost face reg:', (tb - ta).total_seconds())
        # face_reg_cost = tc - tb
        return feat

    def get_sim(self, feat1, feat2):
        return np.dot(feat1, feat2)
