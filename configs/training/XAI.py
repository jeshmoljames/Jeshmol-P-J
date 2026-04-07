# Store gradients and activations
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Attach hooks to last conv layer
target_layer = resnet_model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def detect_objects(frame):
    results = yolo_model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            detections.append({
                "name": r.names[int(box.cls)],
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "confidence": float(box.conf)
            })


  def generate_gradcam(frame):
    global gradients, activations
    gradients.clear()
    activations.clear()

    img = Image.fromarray(frame)
    input_tensor = preprocess(img).unsqueeze(0)

    output = resnet_model(input_tensor)
    class_idx = output.argmax()

    resnet_model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].cpu().numpy()[0]
    acts = activations[0].cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam


    def overlay_xai(frame, cam, detections):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # Draw bounding boxes
    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['name']} {obj['confidence']:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return overlay


    def explain_frame(frame):
    detections = detect_objects(frame)
    cam = generate_gradcam(frame)
    explained = overlay_xai(frame, cam, detections)
    return explained
    return detections
