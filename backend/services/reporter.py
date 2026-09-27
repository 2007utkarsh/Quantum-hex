
import os, json
try:
    from google import genai
except Exception:
    genai=None

def generate_report(data):
    key=os.getenv("GEMINI_API_KEY")
    if not key or genai is None:
        return fallback(data)
    model=os.getenv("GEMINI_MODEL","gemini-3.8-flash")
    client=genai.Client(api_key=key)
    prompt=f"""You are an urban traffic authority intelligence analyst.
Create a decision-ready road and mobility report from the structured platform data below.
Rules:
- Use ONLY supplied facts; do not invent numbers, roads, budgets, causes, or locations.
- Medium length: about 450-650 words, not a one-paragraph summary.
- Include: Executive Summary, Key Road Findings, Safety/Traffic Implications, Priority Actions, Maintenance/Monitoring Recommendations.
- Explain that AI detections are observations requiring appropriate authority verification before physical work.
- If data is simulated, explicitly say so.
DATA:
{json.dumps(data, default=str)}"""
    try:
        response=client.models.generate_content(model=model,contents=prompt)
        return response.text or fallback(data)
    except Exception as e:
        return fallback(data)+f"\n\n[Gemini unavailable: {type(e).__name__}]"

def fallback(data):
    ds=data.get("detections",[])
    high=sum(1 for x in ds if x.get("severity")=="HIGH")
    roads={}
    for x in ds:
        roads[x.get("road","Unknown")]=roads.get(x.get("road","Unknown"),0)+1
    top=", ".join(f"{k} ({v})" for k,v in sorted(roads.items(),key=lambda x:-x[1])[:5]) or "No road detections recorded."
    return f"""Executive Summary

The Urban Intelligence Platform currently contains {len(ds)} stored AI road-defect observations, including {high} marked high severity. The observations are designed to help a traffic authority focus inspection and maintenance resources on corridors showing repeated or high-impact road problems.

Key Road Findings

The most represented roads in the stored detection data are: {top}. Each AI observation includes a location, time, bus/camera source and confidence value where available. Repeated observations from different buses can be aggregated into a single road event to reduce duplicate tickets.

Safety and Traffic Implications

Road defects can contribute to slower traffic, uncomfortable bus movement and increased risk for road users. The platform can combine road-condition signals with fleet telemetry and community reports to identify corridors that deserve closer inspection. These AI observations should be treated as decision-support evidence rather than an automatic repair order.

Priority Actions

Traffic authorities should first inspect high-severity or repeatedly observed defects, verify the physical condition on site, and then assign a maintenance priority. GIS visualization can help compare affected corridors and coordinate field teams.

Maintenance and Monitoring Recommendations

After verification, create a work order and monitor its status through completion. Continue collecting observations from buses so that repaired locations can be checked for recurrence. Community submissions can provide an additional signal between fleet passes.

Note: This report is generated from currently stored platform data. No unsupported statistics or claims have been added."""

def generate_video_damage_report(video, detections):
    """Decision-ready report for one bus video. Facts are deterministic; Gemini only polishes wording."""
    ds=detections or []
    n=len(ds)
    avg=(sum(float(x.get('confidence',0)) for x in ds)/n) if n else 0
    max_conf=max((float(x.get('confidence',0)) for x in ds), default=0)
    max_area=max((float(x.get('bbox_area_ratio',0))*100 for x in ds), default=0)
    max_len=max((float(x.get('mask_length_px',0)) for x in ds), default=0)
    max_breadth=max((float(x.get('mask_breadth_px',0)) for x in ds), default=0)
    total_hits=sum(int(x.get('track_hits') or 0) for x in ds)
    high=sum(1 for x in ds if x.get('severity')=='HIGH')
    medium=sum(1 for x in ds if x.get('severity')=='MEDIUM')
    duration=float(video.get('duration_sec') or 0)
    # A transparent, reproducible priority heuristic. It is explicitly not a physical
    # engineering measurement: pixel geometry needs calibration/depth to become metres.
    persistence=(total_hits/n if n else 0)
    score=(min(35,high*12)+min(20,medium*7)+min(20,persistence*2)+min(15,max_area*2)+min(10,max_conf/10))
    priority='CRITICAL' if score>=75 else 'HIGH' if score>=48 else 'MEDIUM' if score>=24 else 'LOW'
    facts={
      'duration_sec':round(duration,2),'physical_potholes':n,'high_severity':high,'medium_severity':medium,
      'average_confidence_pct':round(avg,1),'maximum_confidence_pct':round(max_conf,1),
      'largest_bbox_area_pct_of_image':round(max_area,2),'largest_mask_length_px':round(max_len,1),
      'largest_mask_breadth_px':round(max_breadth,1),'total_confirmed_track_hits':total_hits,
      'priority_score':round(score,1),'recommended_priority':priority,
      'raw_detections':video.get('raw_detection_count',0),'processed_frames':video.get('processed_frames',0),
      'frame_stride':video.get('frame_stride',1),'zone_detection_count':video.get('zone_detection_count',0),
      'zone_timestamp_sec':video.get('zone_timestamp_sec'),
    }
    key=os.getenv('GEMINI_API_KEY')
    if key and genai is not None and os.getenv('GEMINI_VIDEO_REPORT','true').lower()=='true':
        try:
            model=os.getenv('GEMINI_MODEL','gemini-3.8-flash')
            client=genai.Client(api_key=key)
            prompt=f"""Create a concise but professional road-damage assessment for a traffic authority from ONLY these measured facts. Do not invent metres, causes, budgets, traffic counts, or locations. Clearly distinguish AI observation from field verification. Include: Executive assessment, Damage inventory, Evidence and persistence, Priority recommendation, Field action. Explain that pixel dimensions are image-space only unless camera calibration/depth exists. FACTS: {json.dumps(facts,default=str)}"""
            response=client.models.generate_content(model=model,contents=prompt)
            if response.text:
                return response.text
        except Exception:
            pass
    return f"""EXECUTIVE ASSESSMENT\n\nThe uploaded bus-camera video is {facts['duration_sec']} seconds long. FleetNetra retained {n} physical pothole track(s) after temporal confirmation from {facts['raw_detections']} frame-level detections. The automated assessment recommends {priority} priority for field inspection.\n\nDAMAGE INVENTORY\n\n• Physical pothole tracks retained: {n}\n• High-severity observations: {high}\n• Medium-severity observations: {medium}\n• Average detection confidence: {avg:.1f}%\n• Maximum detection confidence: {max_conf:.1f}%\n• Confirmed track hits: {total_hits}\n• Maximum observed bounding-box area: {max_area:.2f}% of image\n\nEVIDENCE AND PERSISTENCE\n\nThe system processed {facts['processed_frames']} sampled frames using a stride of {facts['frame_stride']}. The strongest simultaneous road-zone evidence contains {facts['zone_detection_count']} detection(s) at video timestamp {facts['zone_timestamp_sec'] if facts['zone_timestamp_sec'] is not None else 'not available'} seconds. Each retained physical track has a context frame and a detail evidence frame.\n\nSIZE NOTE\n\nVisible dimensions are reported in image pixels. They should not be interpreted as metres without camera calibration, camera geometry and/or depth information.\n\nPRIORITY RECOMMENDATION\n\nAutomated priority: {priority} (transparent prototype score: {score:.1f}/100). This is a decision-support signal, not an automatic repair order.\n\nFIELD ACTION\n\nInspect the retained damage locations using the wide zone evidence first, then the context/detail evidence for each track. Confirm the physical defect, extent and safety impact on site before issuing maintenance work. Future bus observations of the same road zone should be used to confirm recurrence and update priority."""

def generate_fast_video_damage_report(video, detections):
    """Deterministic, low-latency report used on the edge ingest path.
    It never calls an LLM, so report creation cannot block video ingestion."""
    ds=detections or []
    n=len(ds)
    avg=sum(float(x.get('confidence',0)) for x in ds)/n if n else 0
    max_conf=max((float(x.get('confidence',0)) for x in ds),default=0)
    high=sum(1 for x in ds if str(x.get('severity','')).upper()=='HIGH')
    medium=sum(1 for x in ds if str(x.get('severity','')).upper()=='MEDIUM')
    low=sum(1 for x in ds if str(x.get('severity','')).upper()=='LOW')
    hits=sum(int(x.get('track_hits') or 0) for x in ds)
    max_area=max((float(x.get('bbox_area_ratio') or 0)*100 for x in ds),default=0)
    max_len=max((float(x.get('mask_length_px') or 0) for x in ds),default=0)
    max_breadth=max((float(x.get('mask_breadth_px') or 0) for x in ds),default=0)
    persistence=(hits/n if n else 0)
    raw=int(video.get('raw_detections') or video.get('raw_detection_count') or 0)
    frames=int(video.get('processed_frames') or 0)
    zones=int(video.get('zone_detection_count') or 0)
    score=min(100, min(35,high*12)+min(20,medium*7)+min(20,persistence*2)+min(15,max_area*2)+min(10,max_conf/10))
    priority='CRITICAL' if score>=75 else 'HIGH' if score>=48 else 'MEDIUM' if score>=24 else 'LOW'
    facts={
      'video_id':video.get('video_id'),'duration_sec':round(float(video.get('duration_sec') or 0),2),
      'physical_potholes':n,'raw_detections':raw,'processed_frames':frames,
      'frame_stride':video.get('frame_stride',1),'high_severity':high,'medium_severity':medium,'low_severity':low,
      'average_confidence_pct':round(avg,1),'maximum_confidence_pct':round(max_conf,1),
      'confirmed_track_hits':hits,'largest_bbox_area_pct':round(max_area,2),
      'largest_mask_length_px':round(max_len,1),'largest_mask_breadth_px':round(max_breadth,1),
      'zone_detection_count':zones,'zone_timestamp_sec':video.get('zone_timestamp_sec'),
      'priority_score':round(score,1),'recommended_priority':priority
    }
    zone_ts=f"{facts['zone_timestamp_sec']:.2f}s" if isinstance(facts['zone_timestamp_sec'],(int,float)) else 'not available'
    return f"""EXECUTIVE ASSESSMENT\n\nFleetNetra automatically analyzed the uploaded bus-camera segment ({facts['duration_sec']} seconds) and retained {n} physically tracked road-defect observation(s) from {raw} frame-level detections. The current automated recommendation is {priority} priority for field inspection. This report was drafted automatically at ingest and sent to the Traffic Authority workspace.\n\nDAMAGE INVENTORY\n\n• Physical pothole tracks retained: {n}\n• High / medium / low observations: {high} / {medium} / {low}\n• Average / maximum confidence: {avg:.1f}% / {max_conf:.1f}%\n• Confirmed track hits: {hits}\n• Largest observed bounding-box area: {max_area:.2f}% of image\n{f'• Largest segmented visible shape: {max_len:.0f} × {max_breadth:.0f} px' if max_len and max_breadth else '• Segmentation dimensions: not available on the low-latency ingest path'}\n\nEVIDENCE & PERSISTENCE\n\nThe pipeline processed {frames} sampled frames at stride {facts['frame_stride']}. The strongest simultaneous road-zone evidence contains {zones} detection(s) at approximately {zone_ts}. Each retained track keeps its own context/detail evidence so an officer can inspect the defect without losing the surrounding road context.\n\nSEVERITY & PRIORITY\n\nAutomated priority: {priority} (transparent prototype score {score:.1f}/100). The score combines observed severity, persistence, confidence and visible image-space extent. It is a decision-support signal, not an engineering measurement or automatic repair order.\n\nFIELD ACTION\n\nTraffic Authority should review the wide zone evidence first, then the context/detail evidence for each retained track. Confirm physical extent and safety impact on site before maintenance. Pixel dimensions must not be treated as metres without camera calibration and/or depth. Repeated observations from additional buses can strengthen confidence in the same road-damage zone.\n\nREPORT DELIVERY\n\nStatus: SENT TO AUTHORITY · Source: BUS EDGE AI · Recipient: TRAFFIC AUTHORITY""", facts
