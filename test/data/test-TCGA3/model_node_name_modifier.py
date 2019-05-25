import onnx
import sys

from onnx.onnx_pb import ModelProto

proto = ModelProto()

f = open('TCGA3.onnx', "rb")
proto.ParseFromString(f.read())
f.close()

nodes = proto.graph.node
index = 0
for node in nodes:
    node.name = 'node' + str(index)
    index = index + 1
f = open(r"TCGA3_modified.onnx", "wb")
f.write(proto.SerializeToString())
f.close()
