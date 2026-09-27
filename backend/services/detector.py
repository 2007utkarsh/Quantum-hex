from ultralytics import YOLO

# SAM is loaded lazily because segmentation is only needed for retained best
# evidence frames, not for every video frame. The existing YOLO detector remains
# the fast first-pass detector.
_SAM=None
from PIL import Image
import io, os, time, uuid, tempfile
import cv2
import numpy as np

MODEL_URL = os.getenv("POTHOLE_MODEL_URL","https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt")
MODEL_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","models"))
MODEL_PATH=os.path.join(MODEL_DIR,"pothole_best.pt")
_model=None

def get_model():
    global _model
    if _model is None:
        os.makedirs(MODEL_DIR,exist_ok=True)
        _model=YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_URL)
    return _model

def get_segmenter():
    """Load an Ultralytics SAM model only when segmentation is enabled.

    SAM is used as a best-evidence postprocessor. It is intentionally not run on
    every sampled frame because that would add substantial latency on CPU.
    """
    global _SAM
    if os.getenv("POTHOLE_SEGMENTATION", "false").lower() != "true":
        return None
    if _SAM is None:
        from ultralytics import SAM
        model_name=os.getenv("POTHOLE_SEGMENT_MODEL", "sam_b.pt")
        _SAM=SAM(model_name)
    return _SAM

def _segment_evidence(frame, bbox):
    """Segment the detected region on a retained evidence frame.

    This is a segmentation-assisted prototype: the pothole detector supplies a
    prompt box and SAM refines the visible shape. It does NOT claim that the
    resulting mask is a centimetre-accurate physical measurement. A pothole-
    specific segmentation checkpoint can replace SAM later without changing
    the rest of the pipeline.
    """
    if os.getenv("POTHOLE_SEGMENTATION", "false").lower() != "true":
        return None
    try:
        segmenter=get_segmenter()
        if segmenter is None:
            return None
        h,w=frame.shape[:2]
        x1,y1,x2,y2=[float(v) for v in bbox]
        # Give the prompt a small margin so an elongated pothole is less likely
        # to be clipped exactly at the detector's box boundary.
        mx=max(8.0,(x2-x1)*float(os.getenv("POTHOLE_SEGMENT_MARGIN", "0.18")))
        my=max(8.0,(y2-y1)*float(os.getenv("POTHOLE_SEGMENT_MARGIN", "0.18")))
        prompt=[[max(0,x1-mx),max(0,y1-my),min(w-1,x2+mx),min(h-1,y2+my)]]
        result=segmenter.predict(source=frame,bboxes=prompt,verbose=False)[0]
        if result.masks is None or result.masks.data is None:
            return None
        masks=result.masks.data.cpu().numpy()
        if len(masks)==0:
            return None
        # Pick the mask with the greatest overlap with the detector box. This
        # avoids a large unrelated road/background segment when SAM returns
        # more than one candidate.
        best=None; best_score=-1
        bx1,by1,bx2,by2=map(int,[max(0,x1),max(0,y1),min(w,x2),min(h,y2)])
        box_area=max(1,(bx2-bx1)*(by2-by1))
        for m in masks:
            m=cv2.resize(m.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
            inside=int(m[by1:by2, bx1:bx2].sum()) if bx2>bx1 and by2>by1 else 0
            score=inside/box_area
            if score>best_score:
                best_score=score; best=m
        if best is None or int(best.sum())<25:
            return None
        # Keep only the connected component with the strongest overlap with the
        # detector box. This makes the output more stable for irregular road
        # texture around the pothole.
        n, labels, stats, cents=cv2.connectedComponentsWithStats(best.astype(np.uint8),8)
        if n>1:
            best_label=max(range(1,n), key=lambda i: (stats[i,cv2.CC_STAT_AREA] * (1.0 if (bx1<=cents[i][0]<=bx2 and by1<=cents[i][1]<=by2) else 0.35)))
            best=(labels==best_label)
        ys,xs=np.where(best)
        if len(xs)<25:
            return None
        pts=np.column_stack([xs,ys]).astype(np.float32)
        rect=cv2.minAreaRect(pts)
        (cx,cy),(rw,rh),angle=rect
        length=max(float(rw),float(rh)); breadth=min(float(rw),float(rh))
        area_px=int(best.sum())
        overlay=frame.copy()
        overlay[best]=((0.0*overlay[best]).astype(np.uint8)+np.array([255,120,0],dtype=np.uint8))
        blended=cv2.addWeighted(frame,0.72,overlay,0.28,0)
        cv2.drawContours(blended,[cv2.boxPoints(rect).astype(np.int32)],0,(255,255,255),2)
        return {
            "mask":best,"mask_area_px":area_px,
            "mask_length_px":round(length,1),"mask_breadth_px":round(breadth,1),
            "mask_area_ratio":round(area_px/max(1,w*h),5),
            "segmentation_model":os.getenv("POTHOLE_SEGMENT_MODEL","sam_b.pt"),
            "segmentation_overlay":blended,
        }
    except Exception:
        # Segmentation is an enhancement, not a reason to fail video inference.
        return None

def _predict(model, frame):
    """Run one cached YOLO inference directly on an OpenCV BGR frame."""
    device=os.getenv("POTHOLE_DEVICE","cpu")
    imgsz=int(os.getenv("POTHOLE_IMGSZ","416"))
    kwargs=dict(
        source=frame,
        conf=float(os.getenv("POTHOLE_CONFIDENCE","0.35")),
        imgsz=imgsz,
        device=device,
        max_det=int(os.getenv("POTHOLE_MAX_DET","12")),
        verbose=False,
    )
    # TensorRT/CUDA deployments can opt into FP16; never use half precision on CPU.
    if os.getenv("POTHOLE_HALF","false").lower()=="true" and device != "cpu":
        kwargs["half"]=True
    return model.predict(**kwargs)[0]

def _iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0.0
    aa=max(1,(ax2-ax1)*(ay2-ay1)); ab=max(1,(bx2-bx1)*(by2-by1))
    return inter/(aa+ab-inter)

def _center(box):
    return ((box[0]+box[2])/2.0,(box[1]+box[3])/2.0)

def _match_score(det, track, frame_w, frame_h, frame_index):
    """Match a moving-camera detection to an existing track.

    IoU alone is brittle for bus footage because perspective makes the same
    pothole change size and move rapidly through the image. We combine IoU
    with predicted-centroid distance, allowing a short detection gap.
    """
    if frame_index-track["last_frame"] > track["max_gap_frames"]:
        return None
    cx,cy=_center(det["bbox"])
    tcx,tcy=_center(track["bbox"])
    dt=max(1,frame_index-track["last_frame"])
    vx,vy=track.get("velocity",(0.0,0.0))
    px=tcx+vx*dt; py=tcy+vy*dt
    distance=((cx-px)**2+(cy-py)**2)**0.5
    diag=max(1.0,(frame_w**2+frame_h**2)**0.5)
    iou=_iou(det["bbox"],track["bbox"])
    # Relative displacement threshold scales with resolution and is deliberately
    # permissive for a forward-facing camera.
    max_distance=max(90.0,diag*0.16)
    if iou < 0.05 and distance > max_distance:
        return None
    center_score=max(0.0,1.0-distance/max_distance)
    return 0.65*center_score+0.35*iou

def _boxes(result):
    out=[]
    if result.boxes is not None:
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist()
        ):
            out.append({
                "bbox":[round(v,1) for v in box],
                "confidence":float(conf),
                "class_id":int(cls)
            })
    return out

def _sharpness_score(frame):
    """Fast image-quality score used to choose the clearest evidence frame."""
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    value=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    # Saturate so one noisy frame cannot dominate the evidence score.
    return min(1.0, value/500.0)

def _evidence_score(confidence, bbox, frame_w, frame_h, sharpness):
    """Rank evidence by confidence, visible pothole area and frame sharpness."""
    x1,y1,x2,y2=bbox
    area=max(0.0,x2-x1)*max(0.0,y2-y1)
    area_ratio=min(1.0, area/max(1.0,frame_w*frame_h)*18.0)
    return 0.55*float(confidence)+0.25*area_ratio+0.20*float(sharpness)

def _context_evidence_score(confidence, bbox, frame_w, frame_h, sharpness):
    """Prefer clear frames where the pothole remains visible but road context is large."""
    x1,y1,x2,y2=bbox
    area=max(0.0,x2-x1)*max(0.0,y2-y1)
    ratio=area/max(1.0,frame_w*frame_h)
    # Best context is neither a tiny speck nor a near full-frame crop.
    visibility=max(0.0,1.0-abs(ratio-0.045)/0.045)
    edge_margin=min(x1,y1,frame_w-x2,frame_h-y2)/max(1.0,min(frame_w,frame_h))
    return 0.45*float(confidence)+0.20*float(sharpness)+0.25*visibility+0.10*max(0.0,min(1.0,edge_margin*8.0))

def _detection_metrics(bbox, frame_w, frame_h):
    x1,y1,x2,y2=bbox
    width=max(0.0,x2-x1); height=max(0.0,y2-y1)
    area=width*height
    image_area=max(1.0,frame_w*frame_h)
    return {
        "bbox_width_px":round(width,1),
        "bbox_height_px":round(height,1),
        "bbox_area_px":round(area,1),
        "bbox_width_ratio":round(width/max(1.0,frame_w),4),
        "bbox_height_ratio":round(height/max(1.0,frame_h),4),
        "bbox_area_ratio":round(area/image_area,5),
    }

def detect_potholes(raw, filename):
    img=Image.open(io.BytesIO(raw)).convert("RGB")
    arr=np.array(img)
    model=get_model()
    t=time.perf_counter()
    r=_predict(model, cv2.cvtColor(arr,cv2.COLOR_RGB2BGR))
    ms=(time.perf_counter()-t)*1000
    detections=_boxes(r)
    annotated=r.plot()
    name=f"{uuid.uuid4().hex}_annotated.jpg"
    path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","evidence",name))
    cv2.imwrite(path, annotated)
    return {"detections":detections,"annotated_path":path,"width":img.width,"height":img.height,"inference_ms":round(ms,1)}

def process_video(raw, filename):
    """Decode video, detect potholes, track them and keep one best evidence frame per physical track."""
    model=get_model()
    suffix=os.path.splitext(filename or "bus.mp4")[1] or ".mp4"
    tmp=None; cap=None
    try:
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f:
            f.write(raw); tmp=f.name
        cap=cv2.VideoCapture(tmp)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            raise ValueError("OpenCV could not open this video. Use MP4/H.264, MOV, or WEBM.")

        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration=(frame_count/fps) if frame_count and fps else 0.0
        stride=max(1,int(os.getenv("POTHOLE_FRAME_STRIDE","2")))
        max_seconds=float(os.getenv("POTHOLE_MAX_VIDEO_SECONDS","0"))
        max_frames=int(max_seconds*fps) if max_seconds>0 else 0
        max_gap_seconds=float(os.getenv("POTHOLE_TRACK_MAX_GAP_SECONDS","3.0"))
        max_gap_frames=max(1,int(fps*max_gap_seconds))
        min_track_hits=max(1,int(os.getenv("POTHOLE_MIN_TRACK_HITS","2")))
        high_conf_single=float(os.getenv("POTHOLE_SINGLE_FRAME_CONFIDENCE","0.70"))

        tracks=[]; frame_index=0; processed=0; inference_ms=0.0; raw_detection_count=0
        zone_best_count=0; zone_best_score=-1.0; zone_frame_path=None; zone_frame_index=None; zone_timestamp_sec=None
        start=time.perf_counter(); video_id=uuid.uuid4().hex[:12].upper()

        while True:
            ok,frame=cap.read()
            if not ok: break
            if max_frames and frame_index>=max_frames: break
            if frame_index % stride != 0:
                frame_index += 1; continue

            t=time.perf_counter(); r=_predict(model,frame)
            inference_ms += (time.perf_counter()-t)*1000; processed += 1
            current=_boxes(r); raw_detection_count += len(current); video_sec=frame_index/fps
            sharpness=_sharpness_score(frame)
            zone_score=(len(current)*0.75)+sharpness*0.25
            if len(current)>0 and (len(current)>zone_best_count or (len(current)==zone_best_count and zone_score>zone_best_score)):
                zone_best_count=len(current); zone_best_score=zone_score
                zone_frame_index=frame_index; zone_timestamp_sec=round(video_sec,2)
                zone_frame_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","evidence",f"{video_id}_zone_evidence.jpg"))
                cv2.imwrite(zone_frame_path,r.plot())

            for d in current:
                best_track=None; best_score=-1.0
                for tr in tracks:
                    score=_match_score(d,tr,width,height,frame_index)
                    if score is not None and score>best_score:
                        best_score=score; best_track=tr

                metrics=_detection_metrics(d["bbox"],width,height)
                ev_score=_evidence_score(d["confidence"],d["bbox"],width,height,sharpness)
                ctx_score=_context_evidence_score(d["confidence"],d["bbox"],width,height,sharpness)

                if best_track is not None and best_score>=0.20:
                    old_center=_center(best_track["bbox"]); new_center=_center(d["bbox"])
                    dt=max(1,frame_index-best_track["last_frame"])
                    best_track["velocity"]=((new_center[0]-old_center[0])/dt,(new_center[1]-old_center[1])/dt)
                    best_track["bbox"]=d["bbox"]
                    best_track["last_frame"]=frame_index
                    best_track["confidence"]=max(best_track["confidence"],d["confidence"])
                    best_track["hits"] += 1
                    best_track["last_timestamp_sec"]=round(video_sec,2)
                    if ctx_score > best_track.get("best_context_score",-1.0):
                        best_track["best_context_score"]=ctx_score
                        best_track["best_context_frame"]=frame_index
                        best_track["context_timestamp_sec"]=round(video_sec,2)
                        context_img=frame.copy()
                        cv2.rectangle(context_img,(int(d["bbox"][0]),int(d["bbox"][1])),(int(d["bbox"][2]),int(d["bbox"][3])),(0,220,255),2)
                        cv2.putText(context_img,f"{best_track["track_id"]}  {d["confidence"]*100:.1f}%",(max(8,int(d["bbox"][0])),max(20,int(d["bbox"][1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),2,cv2.LINE_AA)
                        cv2.imwrite(best_track["context_path"],context_img)
                    if ev_score > best_track["best_evidence_score"]:
                        best_track.update({
                            "best_evidence_score":ev_score,
                            "best_confidence":float(d["confidence"]),
                            "best_frame":frame_index,
                            "best_timestamp_sec":round(video_sec,2),
                            "best_bbox":d["bbox"],
                            "best_metrics":metrics,
                            "best_frame_image":frame.copy(),
                        })
                        annotated=r.plot()
                        cv2.imwrite(best_track["evidence_path"],annotated)
                    continue

                track_no=len(tracks)+1
                track_id=f"T-{track_no:03d}"
                name=f"{uuid.uuid4().hex}_video_track_{track_no}.jpg"
                path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","evidence",name))
                cv2.imwrite(path,r.plot())
                context_name=f"{uuid.uuid4().hex}_video_track_{track_no}_context.jpg"
                context_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","evidence",context_name))
                context_img=frame.copy()
                cv2.rectangle(context_img,(int(d["bbox"][0]),int(d["bbox"][1])),(int(d["bbox"][2]),int(d["bbox"][3])),(0,220,255),2)
                cv2.putText(context_img,f"{track_id}  {d["confidence"]*100:.1f}%",(max(8,int(d["bbox"][0])),max(20,int(d["bbox"][1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),2,cv2.LINE_AA)
                cv2.imwrite(context_path,context_img)
                tracks.append({
                    "track_id":track_id,"bbox":d["bbox"],"last_frame":frame_index,
                    "confidence":float(d["confidence"]),"best_confidence":float(d["confidence"]),
                    "velocity":(0.0,0.0),"hits":1,"max_gap_frames":max_gap_frames,
                    "evidence_path":path,"best_evidence_score":ev_score,
                    "context_path":context_path,"best_context_score":ctx_score,
                    "best_context_frame":frame_index,"context_timestamp_sec":round(video_sec,2),
                    "best_frame":frame_index,"best_frame_image":frame.copy(),"best_timestamp_sec":round(video_sec,2),
                    "last_timestamp_sec":round(video_sec,2),"best_bbox":d["bbox"],
                    "best_metrics":metrics
                })

            # Keep completed tracks for final physical-pothole accounting;
            # _match_score itself rejects tracks outside the active gap window.
            frame_index += 1

        elapsed_ms=(time.perf_counter()-start)*1000
        # Refine only the retained best frame of each confirmed track. This keeps
        # the video path fast while providing a much better estimate of the
        # visible pothole shape than a raw bounding box.
        segmentation_ms=0.0
        segmentation_count=0
        for tr in tracks:
            confirmed=tr["hits"]>=min_track_hits or tr["best_confidence"]>=high_conf_single
            if not confirmed or tr.get("best_frame_image") is None:
                continue
            st=time.perf_counter()
            seg=_segment_evidence(tr["best_frame_image"],tr["best_bbox"])
            segmentation_ms += (time.perf_counter()-st)*1000
            if seg:
                segmentation_count += 1
                seg_name=f"{uuid.uuid4().hex}_video_track_{tr['track_id']}_segmented.jpg"
                seg_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","evidence",seg_name))
                cv2.imwrite(seg_path,seg["segmentation_overlay"])
                tr["segmentation_path"]=seg_path
                tr["segmentation_metrics"]={k:v for k,v in seg.items() if k not in {"mask","segmentation_overlay"}}

        detections=[]
        for tr in tracks:
            # A single high-confidence frame can be useful, but ordinary detections
            # must persist across at least two sampled frames before becoming a case.
            confirmed=tr["hits"]>=min_track_hits or tr["best_confidence"]>=high_conf_single
            if not confirmed:
                continue
            d={
                "bbox":tr["best_bbox"],
                "confidence":tr["best_confidence"],
                "track_id":tr["track_id"],
                "video_id":video_id,
                "frame_index":tr["best_frame"],
                "timestamp_sec":tr["best_timestamp_sec"],
                "annotated_path":tr["evidence_path"],
                "context_path":tr.get("context_path"),
                "context_frame":tr.get("best_context_frame"),
                "context_timestamp_sec":tr.get("context_timestamp_sec"),
                "segmentation_path":tr.get("segmentation_path"),
                "hits":tr["hits"],
                "evidence_score":round(tr["best_evidence_score"],3),
                **tr["best_metrics"],
                **(tr.get("segmentation_metrics") or {})
            }
            detections.append(d)

        return {
            "video_id":video_id,"detections":detections,"width":width,"height":height,
            "fps":round(fps,2),"frame_count":frame_count,"duration_sec":round(duration,2),
            "processed_frames":processed,"frame_stride":stride,"track_count":len(detections),
            "raw_track_count":len(tracks),"inference_ms":round(inference_ms,1),
            "processing_ms":round(elapsed_ms,1),
            "segmentation_ms":round(segmentation_ms,1),
            "segmentation_count":segmentation_count,
            "zone_evidence_path":zone_frame_path,"zone_frame_index":zone_frame_index,"zone_timestamp_sec":zone_timestamp_sec,
            "zone_detection_count":zone_best_count,
            "raw_detection_count":raw_detection_count,
            "confirmed_track_count":len(detections),
            "effective_processing_fps":round(processed/(elapsed_ms/1000),2) if elapsed_ms else 0,
        }
    finally:
        if cap is not None: cap.release()
        if tmp and os.path.exists(tmp): os.remove(tmp)

