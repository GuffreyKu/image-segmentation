import cv2
import torch
import numpy as np
import time
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def decode_segmap(image, decode=[1, 0], nc=2):
    label_colors = np.array([(decode[0], decode[0], decode[0]),  # background
                             (decode[1], decode[1], decode[1]),  # target
                             ])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


model = torch.jit.load('result/model_traced.pt', map_location='cpu')
model = model.to(DEVICE)
model.eval()
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while (True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 360))
        frame = frame.astype(np.float32)/255
        imgTensor = torch.from_numpy(frame.transpose((2, 0, 1)))
        imgTensor = imgTensor.unsqueeze_(0)

        with torch.no_grad():
            inf_time = time.time()
            imgTensor = imgTensor.to(DEVICE)
            preds = model(imgTensor)
            preds = preds.detach().cpu().numpy()
            inf_end_time = time.time()
            print(inf_end_time - inf_time)

            # preds = torch.argmax(preds, axis=1)
            # preds = preds.squeeze()
            # preds = preds.detach().cpu().numpy()

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
