import onnxruntime as ort
import numpy as np
import sys
import traceback
import os

MODEL_PATH = "models/pose_model_final.onnx"
EXPECTED_OUTPUT_DIM = 4 
BATCH_SIZE = 2 

def main():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")

        print(f"Loading ONNX model from {MODEL_PATH}...")
        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        input_shape = sess.get_inputs()[0].shape
        input_shape = [BATCH_SIZE if d is None or d == 'batch' else d for d in input_shape]
        dummy_input = np.random.rand(*input_shape).astype(np.float32)

        print(f"Running inference with dummy input of shape {dummy_input.shape}...")
        output = sess.run([output_name], {input_name: dummy_input})
        output = output[0]

        print(f"Model output shape: {output.shape}")

        assert output.ndim == 2, f"Expected 2D output, got {output.ndim}D"
        assert output.shape[1] == EXPECTED_OUTPUT_DIM, f"Expected output dim {EXPECTED_OUTPUT_DIM}, got {output.shape[1]}"

        if not np.all(np.abs(output) <= 1.0):
            raise ValueError("Output values out of expected range [-1, 1]")

        print("Sanity check PASSED ✅")
        sys.exit(0)

    except Exception as e:
        print("Sanity check FAILED ❌")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()