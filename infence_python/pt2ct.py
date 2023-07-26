import torch
import coremltools as ct
import numpy as np

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == '__main__':
    X = torch.rand(1, 3, 360, 640)

    traced_model = torch.jit.load('result/model_traced.pt', map_location='cpu')
    traced_model = traced_model.to(DEVICE)
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input",
                              shape=X.shape,
                              dtype=np.float16)],
        outputs=[ct.TensorType(name="output",
                               dtype=np.float32)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13
    )
    print(model)
    model.save("result/model.mlpackage")
