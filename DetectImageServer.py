#!/usr/bin/env python


from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
from collections import namedtuple
import tensorflow as tf
from PIL import Image
from io import BytesIO as bytesIO
import os

HttpResponse = namedtuple('HttpResponse', 'status data')
level_matcher_w=240
level_matcher_h=150
file_directory="/home/prasad/workspace/summonerswar/"

actionmapper={
    "victory":0, "defeated":1, "levelup":2, "maxlevel":3,
    "5starrune":4, "home":5, "play":6, "replaybutton":7,
    "revivebutton":8, "rune":9, "startbutton":10,"cleardungeon":11, "pause":12,
    "okbutton":13
}
"""
victory levelup = 5
victory maxlevel = 9
defeated levelup = 6
defeated maxlevel = 10
"""


width=420
height=150
image_matcher =[(10, 950, 10 +420 , 950 + 150),
                (805,815, 805 + 420, 815 + 150),
                (645,800, 645 + 420, 800 + 150),
                (655, 90, 655 + 420, 90 + 150),
                (745, 580, 745 + 420, 580 + 150),
                (600, 90, 600 + 420, 90 + 150),
                (540, 640, 540+ 420, 640 + 150),
                (1000, 890, 1000 + 420, 890 + 150),
                (540, 640, 540 + 420, 640 + 150),
                (500,510, 500 + 420, 510 + 150),
                (1480,680, 1480 + 420, 680 + 150),
                (995, 90, 995 + 420, 90 + 150),
                (875,600, 875+level_matcher_w, 600 + level_matcher_h),
                (1270, 600, 1270+level_matcher_w, 600 + level_matcher_h),
                (480,770,480+level_matcher_w, 770 + level_matcher_h)]

max_level_map = [(875,600, 875+level_matcher_w, 600 + level_matcher_h),
                (1270, 600, 1270+level_matcher_w, 600 + level_matcher_h),
                (480,770,480+level_matcher_w, 770 + level_matcher_h)]                

class DetectImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print ( self.path)
        if ( self.path.startswith('/detect/image/')):
            response = self.detectimage()
            self.serve(response)
            return
        if (self.path.startswith('/detect/maxlevel')):
            response = self.maxlevelimage()
            self.serve(response)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write('hello world')
        return
    def serve(self, response):
        self.send_response(response.status)
        self.end_headers()
        self.wfile.write(response.data)
        return
    def detectimage(self):
        imagename = self.path.split("/")[-1]
        print ("----"+imagename)
        if os.path.isfile(file_directory + imagename + ".jpg") is False:
            res = HttpResponse(status=404, data="Not found")
            return res
        images = classify_image(imagename)
        image_str = '';
        result=0
        print ("------------")
        for i in images:
            print ("--" + str(actionmapper[i]))
            result |=(1<< actionmapper[i])
        res = HttpResponse(status=200, data=str(hex(result)))
        return res
    def maxlevelimage(self):
        imagename = self.path.split("/")[-1]
        if os.path.isfile(file_directory + imagename + ".jpg") is False:
            res = HttpResponse(status=404, data="Not found")
            return res
        result=find_max_image(imagename)
        res = HttpResponse(status=200, data=result)
        return res
        


label_lines = [line.rstrip() \
               for line in \
               tf.gfile.GFile("./tf_files/retrained_labels.txt")]

with tf.gfile.FastGFile("./tf_files/retrained_graph.ph", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _=tf.import_graph_def(graph_def, name='')


def crop_image(imagename, arg0=image_matcher):
    ximage = Image.open(file_directory + imagename + ".jpg").convert('L')
    def func_crop(arg):
        tmpimage = ximage.crop(arg)
        f = bytesIO()
        tmpimage.save(f, format='JPEG')
        return f.getvalue()
    images = map(func_crop, arg0)
    return images

def classify_image(imagename):
    results = []
    images = list(crop_image(imagename))
    with tf.Session() as session:
        softmax_tensor = session.graph.get_tensor_by_name('final_result:0')
        for image in images:
            predictions = session.run(softmax_tensor, {'DecodeJpeg/contents:0':image})
            top_k=predictions[0].argsort()[-len(predictions[0]):][::-1]
            scores = ""
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                results.append((human_string, score))
                break
    results.sort(key=lambda x: x[1])
    classify_set = set()
    for rslt in results:
        print (rslt)
    absresults = list(filter(lambda x : x[1] > 0.9, results))
    if ( len(absresults) == 0):
        print("1 result found 0")
        absresults = list(filter(lambda x: x[0] == "victory" and x[1] > 0.7, results))
    if (len(absresults) == 0):
        print ("2 result found 0")
        classify_set.add(results[-1][0])
    for i in absresults:
        classify_set.add(i[0])
    return classify_set


def find_max_image(imagename):
    images = list(crop_image(imagename, max_level_map))
    result = 0
    binarr = [0, 1, 2]
    countbinarr=0
    with tf.Session() as session:
        softmax_tensor = session.graph.get_tensor_by_name('final_result:0')
        for image in images:
            predictions = session.run(softmax_tensor, {'DecodeJpeg/contents:0':image})
            top_k=predictions[0].argsort()[-len(predictions[0]):][::-1]
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print ("%s - %s" % (human_string, str(score)))
                if human_string == "maxlevel":
                    result |= (1 << binarr[countbinarr])
                break
            countbinarr = countbinarr + 1
    print ( bin(result))
    return str(result)
            



if __name__ == '__main__':
    server = HTTPServer(('localhost', 9009), DetectImageHandler)
    print ("Starting server ....")
    server.serve_forever()
