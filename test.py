from mtcnn import caffe_pb2 as pb


net = pb.NetParameter()

with open('./mtcnn/det1.caffemodel', 'rb') as f:
    net.ParseFromString(f.read())

