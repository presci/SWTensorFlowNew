#!/bin/python


import tensorflow as tf
from PIL import Image
from io import BytesIO as bytesIO

width=420
height=150


image_matcher =[(10, 950, 10 +420 , 950 + 150),
    (805,815, 805 + 420, 815 + 150),
    (645,800, 645 + 420, 800 + 150),
    ( 655, 90, 655 + 420, 90 + 150),
    (745, 580, 745 + 420, 580 + 150),
    (600, 90, 600 + 420, 90 + 150),
    (540, 640, 540+ 420, 640 + 150),
    (1000, 890, 1000 + 420, 890 + 150),
    (540, 640, 540 + 420, 640 + 150),
    (500,510, 500 + 420, 510 + 150),
    (1480,680, 1480 + 420, 680 + 150)]


def cropimage(imagestr, match=None):
    coordinates = image_matcher[0]
    if coordinates is None:
        return
    image = Image.open(imagestr).convert("L")
    tmpimage = image.crop(coordinates)
    f = bytesIO()
    tmpimage.save(f, format='JPEG')
    return f.getvalue()

#original = Image.open("victory01.jpg").convert("L")
#victoryimage = cropimage(original,655, 90)
#print (victoryimage)


label_lines = [line.rstrip() for line in tf.gfile.GFile("./tf_files/retrained_labels.txt")]

with tf.gfile.FastGFile("./tf_files/retrained_graph.ph", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _=tf.import_graph_def(graph_def, name='')

ximage = Image.open("pause01.jpg").convert("L")
def func_crop(arg):
    tmpimage = ximage.crop(arg)
    f = bytesIO()
    tmpimage.save(f, format='JPEG')
    return f.getvalue()


images = map(func_crop, image_matcher)
results = []

with tf.Session() as session:
    softmax_tensor = session.graph.get_tensor_by_name('final_result:0')
    for cropped_image in list(images):
        predictions = session.run(softmax_tensor, {'DecodeJpeg/contents:0':cropped_image})
        top_k=predictions[0].argsort()[-len(predictions[0]):][::-1]
        scores = "";
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            results.append((human_string, score))
            break
            #print ("%s %s"% (human_string, str(score)))
    results.sort(key=lambda x: x[1])
    for i in results:
        print ( i)
    print ("----------------------------")
    absresults = list(filter(lambda x : x[1] > 0.9, results))
    for i in absresults:
        print (i)
