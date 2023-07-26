import cv2
import numpy as np
import time
import coremltools as ct
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
font = cv2.FONT_HERSHEY_SIMPLEX

model = ct.models.MLModel("result/model.mlpackage",
                          compute_units=ct.ComputeUnit.ALL)

# model = torch.jit.load('result/model_traced.pt', map_location='cpu')
# model = model.to(DEVICE)
# model.eval()


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


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    h = 360
    w = 640
    while (True):
        # 擷取影像
        ret, frame = cap.read()
        if ret is not True:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()
        start = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))

        raw_img = frame
        gausBlur = cv2.GaussianBlur(frame, (55, 55), 0)
        frame = frame/255
        frame = frame.astype(np.float16)
        # frame = Image.fromarray(frame)
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)

        inf_time = time.time()
        coreml_out_dict = model.predict({"input": frame})
        inf_end_time = time.time()
        print(inf_end_time - inf_time)

        preds = coreml_out_dict['output']
        preds = np.argmax(preds, axis=1)
        preds = preds.squeeze()

        end = time.time()
        fps = str(int(1 / (end - start)))

        # print("fps : ", 1/ (end - start))

        predbg = decode_segmap(preds, [1, 0], 2)
        predta = decode_segmap(preds, [0, 1], 2)

        vis = (raw_img * predta) + (gausBlur * predbg)

        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        vis = cv2.resize(vis, (1280, 720))

        cv2.putText(vis, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        # frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow('image', vis)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
