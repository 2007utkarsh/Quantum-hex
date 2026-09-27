# FleetNetra AI — Role-Based Urban Intelligence Prototype

This version adds role-based access for **Traffic Authority, Traffic Inspector, and Citizen** while keeping the bus/edge AI as a machine-to-machine source rather than a human dashboard user.

## Demo accounts
- Authority: `authority` / `authority123`
- Inspector: `inspector` / `inspector123`
- Citizen: `citizen` / `citizen123`

## Run backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python seed_demo.py
python -m uvicorn main:app --reload --port 8000
```

## Run frontend
In another terminal:
```bash
npm install
npm run dev
```

## Role model
- **Authority:** full GIS, fleet intelligence, clustered AI road cases, heatmaps, road priority, assignments, community signals and Gemini reports.
- **Inspector:** assigned GIS/workbench and field verification only.
- **Citizen:** public reporting, own report tracking and safer-route utility.
- **Bus/Edge AI:** no human login; telemetry uses a device API key endpoint.

The prototype uses signed demo session tokens and backend authorization checks. Replace demo credentials and `AUTH_SECRET` before any real deployment.

## Prototype demo flow

- Authority: `authority / authority123`
- Inspector: `inspector / inspector123`
- Citizen: `citizen / citizen123`
- Bus Edge AI Console: open `/bus-demo` (device-authenticated prototype simulator; no human login)

The bus console sends live prototype GPS telemetry with an `X-Device-Key` and sends pothole inference events through `/api/edge/detect`. Authority users consume those stored observations as clustered road cases; authority users do not run the onboard detector.


## Raw bus video → OpenCV frames → YOLO
The Bus / Edge AI Console now accepts **raw bus-camera video** instead of only single images. The backend endpoint `/api/edge/detect-video` uses OpenCV `VideoCapture` to decode the video, samples frames (default: every 3rd frame), runs the already-cached YOLO pothole model directly on OpenCV frames, and sends only temporally de-duplicated pothole observations to the central platform.

### Low-latency choices
- YOLO model is loaded once at backend startup and reused for every frame.
- OpenCV handles video decoding; no PIL conversion is used in the video path.
- Frame sampling (`POTHOLE_FRAME_STRIDE=2`) avoids running inference on every frame.
- Only new pothole observations save an annotated JPEG; the system does not encode a second full annotated video.
- Optional `POTHOLE_HALF=true` can be enabled on CUDA/TensorRT-style deployments.
- Increase stride for faster processing; reduce it to `1` when maximum temporal coverage is more important than latency.

The upload workflow is a demo of the onboard pipeline. For a production bus camera, the same processing function should be connected to a live RTSP/WebRTC/camera stream rather than waiting for a whole file upload.

### Video evidence + road-damage clustering
The bus video pipeline now separates **frame detections**, **physical pothole tracks**, and **operational road cases**. A single uploaded clip can contain multiple physical potholes, but those tracks are grouped into one road-damage cluster for authority workflow instead of generating one case per frame/track. Each physical pothole keeps its best evidence frame selected using confidence, visible bounding-box area, and frame sharpness.

Each retained detection also exposes image-space measurements:
- `bbox_width_px` and `bbox_height_px`
- bounding-box pixel area and image-area ratio
- track hit count (number of sampled frames supporting the track)
- evidence score

These are **pixel/image-space dimensions**. Real-world metres require camera calibration/depth/road-plane scale and are intentionally not fabricated by the prototype.

Useful environment settings:
- `POTHOLE_MIN_TRACK_HITS=2`
- `POTHOLE_SINGLE_FRAME_CONFIDENCE=0.70`
- `POTHOLE_CONFIDENCE=0.35` (candidate threshold; temporal confirmation filters weak one-frame detections)

### Shape-aware pothole evidence
The video pipeline keeps the fast YOLO first pass and temporal tracking, then optionally runs Ultralytics SAM only on each retained best evidence frame. This refines the visible pothole shape and reports mask area, major-axis length and minor-axis breadth in pixels. It does not fabricate metre-scale dimensions; real-world dimensions require camera calibration/depth. The default `sam_b.pt` checkpoint is a general segmentation model. For production/SIH validation, a pothole-specific segmentation checkpoint trained with polygon masks is recommended.

Relevant environment variables:
- `POTHOLE_SEGMENTATION=false`
- `POTHOLE_SEGMENT_MODEL=sam_b.pt`
- `POTHOLE_SEGMENT_MARGIN=0.18`


### Evidence views
Video detections now expose three evidence levels:
- **Zone view:** the widest full-frame road context available, selected from the sampled frame containing the most simultaneous detections. This is the primary evidence for a road-damage cluster.
- **Context view:** a clear frame around each physical pothole while retaining substantial surrounding road.
- **Detail view:** the highest-scoring close detection frame, with optional segmentation refinement.

The backend also keeps completed tracks after they leave the active matching window, so an early pothole is not accidentally discarded before final physical-pothole accounting.


## Low-latency video + damage reporting

The edge path is optimized for continuous raw bus-video flow: cached YOLO weights, OpenCV decoding, a small inference image size, configurable frame sampling, limited detections per frame, and no SAM segmentation on the hot path. Segmentation is an evidence/report enhancement and is disabled by default for live latency. On CUDA, use `POTHOLE_FRAME_STRIDE=1`, `POTHOLE_HALF=true`, and an appropriate device.

The video result separates **raw frame detections**, **confirmed physical tracks**, and **operational road cases**. This prevents every YOLO box from becoming a maintenance ticket.

The Bus/Edge console can generate a decision-ready damage report after processing. Report generation is deliberately separate from edge inference so an LLM never blocks the low-latency detection path.

Assignments are case-level: for a video upload, one assignment covers all physical tracks belonging to that video, and their workflow status is synchronized to the Regional Road Officer queue.
