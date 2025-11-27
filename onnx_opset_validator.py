import onnx
from onnx import checker, helper

model_path = "model.onnx"
model = onnx.load(model_path)

# check for structural issues
try:
    checker.check_model(model)
    print("ONNX model is structurally valid")
except onnx.checker.ValidationError as e:
    print("ONNX model has issues:", e)

# list all operators
ops = set([node.op_type for node in model.graph.node])
print("Ops used in model:", ops)