import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.75
)

# Smoothing / tracking params
ALPHA = 0.2               # lower = smoother, less jitter
MIN_FACE_SIZE = 120
VISIBILITY_MARGIN = 0.02
CENTER_TOLERANCE = 0.20
IOU_MATCH_THRESH = 0.45
MAX_MISSES = 5

# Change source here: 0 = webcam, or path to video file
SOURCE = 0  # e.g. "sample_video.mp4" for a file
cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

next_face_id = 0
tracked = []
frame_idx = 0

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0

def is_fully_visible(box, frame_w, frame_h, margin_frac):
    x, y, w, h = box
    margin_x = int(frame_w * margin_frac)
    margin_y = int(frame_h * margin_frac)
    return (x >= margin_x) and (y >= margin_y) and (x + w <= frame_w - margin_x) and (y + h <= frame_h - margin_y)

def is_centered(box, frame_w, frame_h, tol_frac):
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    center_x = frame_w / 2
    center_y = frame_h / 2
    return (abs(cx - center_x) <= frame_w * tol_frac) and (abs(cy - center_y) <= frame_h * tol_frac)

print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    detections = []
    if results.detections:
        for det in results.detections:
            score = float(det.score[0])
            if score < 0.75:
                continue
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                continue
            detections.append((x, y, width, height, score))

    # Match detections to tracked faces
    matches = {}
    used_tracks = set()
    for di, det in enumerate(detections):
        best_iou = 0.0
        best_t = None
        det_box = det[:4]
        for ti, t in enumerate(tracked):
            if ti in used_tracks:
                continue
            cur_iou = iou(det_box, t['box'])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_t = ti
        if best_iou >= IOU_MATCH_THRESH and best_t is not None:
            matches[di] = best_t
            used_tracks.add(best_t)

    # Update matched tracks
    for di, ti in matches.items():
        x, y, w_box, h_box, score = detections[di]
        t = tracked[ti]
        t['box'] = (x, y, w_box, h_box)
        sx, sy, sw, sh = t['smooth']
        sx = ALPHA * x + (1 - ALPHA) * sx
        sy = ALPHA * y + (1 - ALPHA) * sy
        sw = ALPHA * w_box + (1 - ALPHA) * sw
        sh = ALPHA * h_box + (1 - ALPHA) * sh
        # round to multiples of 4 to reduce jitter
        t['smooth'] = (round(sx / 4) * 4, round(sy / 4) * 4,
                       round(sw / 4) * 4, round(sh / 4) * 4)
        t['last_seen'] = frame_idx
        t['misses'] = 0
        t['score'] = score

    # Create new tracks
    for di, det in enumerate(detections):
        if di in matches:
            continue
        x, y, w_box, h_box, score = det
        tracked.append({
            'id': next_face_id,
            'box': (x, y, w_box, h_box),
            'smooth': (float(x), float(y), float(w_box), float(h_box)),
            'last_seen': frame_idx,
            'misses': 0,
            'score': score
        })
        next_face_id += 1

    # Increment misses
    for t in tracked:
        if t['last_seen'] != frame_idx:
            t['misses'] += 1
    tracked = [t for t in tracked if t['misses'] <= MAX_MISSES]

    # Qualify faces
    qualifying = []
    for t in tracked:
        sx, sy, sw, sh = t['smooth']
        box_int = (int(sx), int(sy), int(sw), int(sh))
        if is_fully_visible(box_int, w, h, VISIBILITY_MARGIN) and sw >= MIN_FACE_SIZE and sh >= MIN_FACE_SIZE:
            qualifying.append((t, box_int))

    # Display logic
    display_list = []
    if len(qualifying) >= 2:
        display_list = qualifying
    elif len(qualifying) == 1:
        t, box_int = qualifying[0]
        if is_centered(box_int, w, h, CENTER_TOLERANCE):
            display_list = qualifying

    # Draw boxes
    for t, box_int in display_list:
        sx, sy, sw, sh = box_int
        color = (0, 255, 0)
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, 2)
        cv2.putText(frame, f"Face {t['id']} {t.get('score',0):.2f}", (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status = f"{len(display_list)} Face(s) Shown"
    color = (0, 255, 0) if len(display_list) > 0 else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Detection (smoothed)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
