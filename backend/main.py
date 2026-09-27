from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services.detector import detect_potholes, process_video
from services.database import init_db, insert_detection, list_detections, list_assigned_detections, insert_telemetry, list_telemetry, insert_complaint, list_complaints, tick_simulated_telemetry, insert_assignment, list_assignments, update_detection_status, update_complaint_status, insert_idea, list_ideas, vote_idea, insert_video_run, get_video_run, insert_damage_report, list_damage_reports, get_damage_report
from services.reporter import generate_report, generate_video_damage_report, generate_fast_video_damage_report
import os, uuid, json, hmac, hashlib, base64
from datetime import datetime, timezone

app=FastAPI(title='FleetNetra AI API',version='2.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
os.makedirs('evidence',exist_ok=True); os.makedirs('data',exist_ok=True)
app.mount('/evidence',StaticFiles(directory='evidence'),name='evidence')
init_db()

@app.on_event('startup')
def warm_edge_model():
    # Load the YOLO model once when the backend starts so the first bus detection
    # does not pay the model-loading/download cost. Failure here is non-fatal;
    # detection will retry lazily on the first request.
    try:
        from services.detector import get_model
        get_model()
    except Exception as exc:
        print(f'[FleetNetra] Edge model warmup deferred: {exc}')

SECRET=os.getenv('AUTH_SECRET','hackathon-demo-secret-change-me').encode()
USERS={
  'authority': {'password':'authority123','role':'AUTHORITY','name':'Command Authority','initials':'CA'},
  'inspector': {'password':'inspector123','role':'INSPECTOR','name':'Regional Road Officer','initials':'RO'},
  'citizen': {'password':'citizen123','role':'CITIZEN','name':'Citizen User','initials':'CU'},
}

def make_token(username,role):
    payload=json.dumps({'u':username,'r':role},separators=(',',':')).encode()
    body=base64.urlsafe_b64encode(payload).decode().rstrip('=')
    sig=hmac.new(SECRET,body.encode(),hashlib.sha256).hexdigest()
    return body+'.'+sig

def current_user(authorization: str|None = Header(default=None)):
    if not authorization or not authorization.startswith('Bearer '): raise HTTPException(401,'Authentication required.')
    token=authorization[7:]
    try:
        body,sig=token.split('.',1)
        expected=hmac.new(SECRET,body.encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig,expected): raise ValueError()
        payload=json.loads(base64.urlsafe_b64decode(body+'='*((4-len(body)%4)%4)))
        if payload.get('u') not in USERS: raise ValueError()
        return payload
    except Exception: raise HTTPException(401,'Invalid or expired session.')

def require(*roles):
    def dep(user=Depends(current_user)):
        if user['r'] not in roles: raise HTTPException(403,'This workspace is not available for your role.')
        return user
    return dep

class Login(BaseModel): username:str; password:str
class Telemetry(BaseModel): bus_id:str; lat:float; lng:float; speed_kmh:float=0; route:str='Unknown'; status:str='ACTIVE'
class Complaint(BaseModel): issue_type:str; location:str; description:str=''; lat:float|None=None; lng:float|None=None
class Assignment(BaseModel): event_id:str; inspector_id:str='inspector'; case_id:str|None=None
class Verify(BaseModel): event_id:str; status:str; remarks:str=''

@app.get('/api/health')
def health(): return {'status':'ok','model':'YOLOv8 edge pothole detector','database':'SQLite','auth':'RBAC','time':datetime.now(timezone.utc).isoformat()}

@app.post('/api/auth/login')
def login(data:Login):
    u=USERS.get(data.username)
    if not u or data.password!=u['password']: raise HTTPException(401,'Invalid demo credentials.')
    return {'token':make_token(data.username,u['role']),'user':{'username':data.username,'role':u['role'],'name':u['name'],'initials':u['initials']}}

@app.get('/api/auth/me')
def me(user=Depends(current_user)): return {'username':user['u'],'role':user['r'],'name':USERS[user['u']]['name'],'initials':USERS[user['u']]['initials']}

@app.post('/api/edge/detect')
async def edge_detect(file:UploadFile=File(...),bus_id:str=Form('DEMO-BUS-01'),camera:str=Form('FRONT-01'),lat:float=Form(26.8467),lng:float=Form(80.9462),road:str=Form('Lucknow District'),x_device_key:str|None=Header(default=None)):
    if x_device_key != os.getenv('DEVICE_API_KEY','urbanai-bus-device-demo'): raise HTTPException(401,'Invalid bus device key.')
    if not file.content_type or not file.content_type.startswith('image/'): raise HTTPException(400,'Please upload an image.')
    raw=await file.read()
    if len(raw)>15*1024*1024: raise HTTPException(413,'Image too large. Maximum 15 MB.')
    try: result=detect_potholes(raw,file.filename or 'frame.jpg')
    except Exception as e: raise HTTPException(500,f'Model inference failed: {e}')
    created=[]
    for d in result['detections']:
        event_id='P-'+uuid.uuid4().hex[:8].upper(); area=max(0,d['bbox'][2]-d['bbox'][0])*max(0,d['bbox'][3]-d['bbox'][1]); frame_area=max(1,result['width']*result['height'])
        severity='HIGH' if area/frame_area>0.08 or d['confidence']>=0.88 else ('MEDIUM' if d['confidence']>=0.60 else 'LOW')
        event={'id':event_id,'event_type':'POTHOLE','road':road,'bus':bus_id,'camera':camera,'confidence':round(d['confidence']*100,2),'lat':lat,'lng':lng,'severity':severity,'time':datetime.now().strftime('%H:%M:%S'),'timestamp':datetime.now(timezone.utc).isoformat(),'bbox':d['bbox'],'evidence_url':f"/evidence/{os.path.basename(result['annotated_path'])}",'source':'EDGE_AI','status':'DETECTED'}
        insert_detection(event); created.append(event)
    return {'model':'peterhdd/pothole-detection-yolov8','detections':created,'annotated_url':f"/evidence/{os.path.basename(result['annotated_path'])}",'image_width':result['width'],'image_height':result['height'],'inference_ms':result['inference_ms']}


@app.post('/api/edge/detect-video')
async def edge_detect_video(file:UploadFile=File(...),bus_id:str=Form('DEMO-BUS-01'),camera:str=Form('FRONT-01'),lat:float=Form(26.8467),lng:float=Form(80.9462),road:str=Form('Lucknow District'),x_device_key:str|None=Header(default=None)):
    """Process raw bus-camera video with OpenCV -> sampled frames -> cached YOLO."""
    if x_device_key != os.getenv('DEVICE_API_KEY','urbanai-bus-device-demo'):
        raise HTTPException(401,'Invalid bus device key.')
    allowed={'video/mp4','video/quicktime','video/webm','video/x-msvideo','video/mpeg'}
    if not file.content_type or file.content_type not in allowed:
        raise HTTPException(400,'Please upload an MP4, MOV, WEBM, AVI or MPEG video.')
    raw=await file.read()
    max_mb=float(os.getenv('MAX_VIDEO_MB','150'))
    if len(raw)>max_mb*1024*1024:
        raise HTTPException(413,f'Video too large. Maximum {int(max_mb)} MB.')
    try:
        result=process_video(raw,file.filename or 'bus-video.mp4')
    except Exception as e:
        raise HTTPException(500,f'Video processing failed: {e}')

    created=[]
    zone_url=(f"/evidence/{os.path.basename(result['zone_evidence_path'])}" if result.get('zone_evidence_path') else None)
    for d in result['detections']:
        event_id='P-'+uuid.uuid4().hex[:8].upper()
        area=max(0,d['bbox'][2]-d['bbox'][0])*max(0,d['bbox'][3]-d['bbox'][1])
        frame_area=max(1,result['width']*result['height'])
        severity='HIGH' if area/frame_area>0.08 or d['confidence']>=0.88 else ('MEDIUM' if d['confidence']>=0.60 else 'LOW')
        context_url=(f"/evidence/{os.path.basename(d['context_path'])}" if d.get('context_path') else None)
        event={
            'id':event_id,'event_type':'POTHOLE','road':road,'bus':bus_id,'camera':camera,
            'confidence':round(d['confidence']*100,2),'lat':lat,'lng':lng,'severity':severity,
            'time':datetime.now().strftime('%H:%M:%S'),
            'timestamp':datetime.now(timezone.utc).isoformat(),
            'bbox':d['bbox'],
            'evidence_url':f"/evidence/{os.path.basename(d['annotated_path'])}",
            'source':'EDGE_VIDEO_AI','status':'DETECTED',
            'frame_index':d['frame_index'],'video_timestamp_sec':d['timestamp_sec'],'video_id':d.get('video_id',result.get('video_id')),'track_id':d.get('track_id'),
            'bbox_width_px':d.get('bbox_width_px'),'bbox_height_px':d.get('bbox_height_px'),'bbox_area_px':d.get('bbox_area_px'),
            'bbox_width_ratio':d.get('bbox_width_ratio'),'bbox_height_ratio':d.get('bbox_height_ratio'),'bbox_area_ratio':d.get('bbox_area_ratio'),
            'mask_area_px':d.get('mask_area_px'),'mask_length_px':d.get('mask_length_px'),'mask_breadth_px':d.get('mask_breadth_px'),'mask_area_ratio':d.get('mask_area_ratio'),
            'context_evidence_url':context_url,'zone_evidence_url':zone_url,'context_frame_index':d.get('context_frame'),'context_timestamp_sec':d.get('context_timestamp_sec'),
            'segmentation_model':d.get('segmentation_model'),'segmentation_evidence_url':(f"/evidence/{os.path.basename(d['segmentation_path'])}" if d.get('segmentation_path') else None),
            'track_hits':d.get('hits'),'evidence_score':d.get('evidence_score')
        }
        insert_detection(event)
        created.append(event)

    video_run={'video_id':result.get('video_id'),'duration_sec':result.get('duration_sec',0),'processed_frames':result.get('processed_frames',0),'frame_stride':result.get('frame_stride',1),'raw_detections':result.get('raw_detection_count',0),'zone_detection_count':result.get('zone_detection_count',0),'zone_timestamp_sec':result.get('zone_timestamp_sec'),'processing_ms':result.get('processing_ms',0),'inference_ms':result.get('inference_ms',0)}
    insert_video_run(video_run)

    # Fast, deterministic report creation happens immediately after ingest. It
    # does not call Gemini, so report drafting adds negligible latency to the
    # edge request. The stored report becomes the Authority's inbox item.
    fast_report, report_facts = generate_fast_video_damage_report(video_run, created)
    report_id='R-'+uuid.uuid4().hex[:10].upper()
    insert_damage_report({'id':report_id,'video_id':result.get('video_id'),'report_type':'VIDEO_DAMAGE','report_text':fast_report,'facts':report_facts,'status':'SENT_TO_AUTHORITY','generated_at':datetime.now(timezone.utc).isoformat(),'recipient_role':'AUTHORITY','source_bus':bus_id,'road':road})

    return {
        'model':'peterhdd/pothole-detection-yolov8',
        'pipeline':'OpenCV video decode -> frame sampling -> cached YOLO inference',
        'detections':created,
        'report':{'id':report_id,'status':'SENT_TO_AUTHORITY','recipient':'TRAFFIC AUTHORITY','auto_generated':True,'text':fast_report},
        'video':{
            'width':result['width'],'height':result['height'],'fps':result['fps'],
            'frame_count':result['frame_count'],'duration_sec':result['duration_sec'],
            'processed_frames':result['processed_frames'],'frame_stride':result['frame_stride']
        },
        'inference_ms':result['inference_ms'],
        'segmentation_ms':result.get('segmentation_ms',0),'segmentation_count':result.get('segmentation_count',0),
        'processing_ms':result['processing_ms'],
        'effective_processing_fps':result['effective_processing_fps'],'unique_physical_potholes':len(created),'raw_tracks':result.get('raw_track_count',len(created)),'raw_detections':result.get('raw_detection_count',0),'confirmed_tracks':result.get('confirmed_track_count',len(created)),'video_id':result.get('video_id'),'zone_evidence_url':zone_url,'zone_frame_index':result.get('zone_frame_index'),'zone_timestamp_sec':result.get('zone_timestamp_sec'),'zone_detection_count':result.get('zone_detection_count',0)
    }

@app.get('/api/detections')
def detections(user=Depends(require('AUTHORITY','INSPECTOR'))):
    rows=list_detections(200) if user['r']=='AUTHORITY' else list_assigned_detections(user['u'])
    assignments=list_assignments()
    latest={}
    for a in assignments:
        if a.get('target_type','DETECTION')=='DETECTION':
            latest[a['event_id']]=a
    for d in rows:
        a=latest.get(d['id'])
        if a:
            d['assigned_officer']='Regional Road Officer'
            d['assignment_status']=a['status']
    return rows

@app.post('/api/telemetry')
def telemetry(t:Telemetry,x_device_key:str|None=Header(default=None)):
    if x_device_key != os.getenv('DEVICE_API_KEY','urbanai-bus-device-demo'): raise HTTPException(401,'Invalid bus device key.')
    insert_telemetry(t.model_dump()); return {'ok':True}

@app.get('/api/fleet/live')
def fleet_live(user=Depends(require('AUTHORITY'))):
    tick_simulated_telemetry(); return list_telemetry()

@app.post('/api/community')
async def community(issue_type:str=Form(...),location:str=Form(...),description:str=Form(''),file:UploadFile|None=File(default=None),user=Depends(require('CITIZEN'))):
    if not location.strip(): raise HTTPException(400,'Location is required.')
    evidence_url=None
    if file:
        if not file.content_type or file.content_type not in {'image/jpeg','image/png','image/webp'}: raise HTTPException(400,'Evidence must be JPG, PNG or WEBP.')
        raw=await file.read()
        if len(raw)>10*1024*1024: raise HTTPException(413,'Evidence image is too large. Maximum 10 MB.')
        ext={'image/jpeg':'.jpg','image/png':'.png','image/webp':'.webp'}[file.content_type]
        name='community-'+uuid.uuid4().hex+ext
        with open(os.path.join('evidence',name),'wb') as out: out.write(raw)
        evidence_url=f'/evidence/{name}'
    cid='CMP-'+uuid.uuid4().hex[:7].upper()
    insert_complaint({'id':cid,'user_id':user['u'],'issue_type':issue_type,'location':location.strip(),'description':description,'lat':None,'lng':None,'status':'SUBMITTED','created_at':datetime.now(timezone.utc).isoformat()})
    # Keep evidence attached to the complaint without changing the existing schema shape.
    if evidence_url:
        import sqlite3
        c=sqlite3.connect(os.path.abspath(os.path.join(os.path.dirname(__file__),'data','urban.db')))
        c.execute('CREATE TABLE IF NOT EXISTS complaint_evidence (complaint_id TEXT PRIMARY KEY,evidence_url TEXT)')
        c.execute('INSERT OR REPLACE INTO complaint_evidence VALUES(?,?)',(cid,evidence_url)); c.commit(); c.close()
    return {'ok':True,'id':cid,'evidence_url':evidence_url}

@app.get('/api/community')
def community_list(user=Depends(current_user)):
    rows=list_complaints() if user['r'] in ('CITIZEN','AUTHORITY','INSPECTOR') else None
    if rows is None: raise HTTPException(403,'Forbidden')
    try:
        import sqlite3
        c=sqlite3.connect(os.path.abspath(os.path.join(os.path.dirname(__file__),'data','urban.db'))); c.row_factory=sqlite3.Row
        ev={r['complaint_id']:r['evidence_url'] for r in c.execute('SELECT complaint_id,evidence_url FROM complaint_evidence').fetchall()}; c.close()
        assignments_by_case={}
        for a in list_assignments():
            if a.get('target_type')=='CITIZEN_REPORT':
                assignments_by_case[a['event_id']]=a
        for row in rows:
            row['evidence_url']=ev.get(row['id'])
            a=assignments_by_case.get(row['id'])
            if a:
                row['assigned_officer']='Regional Road Officer'
                # The complaint status is the source of truth; expose the assignment too.
                row['assignment_status']=a['status']
    except Exception: pass
    return rows

@app.get('/api/community/ideas')
def community_ideas(user=Depends(current_user)):
    if user['r'] not in ('CITIZEN','AUTHORITY','INSPECTOR'): raise HTTPException(403,'Forbidden')
    return list_ideas()

class Idea(BaseModel): title:str; category:str='Road safety'; description:str; complaint_id:str|None=None

@app.post('/api/community/ideas')
def create_idea(data:Idea,user=Depends(require('CITIZEN'))):
    item={'id':'IDEA-'+uuid.uuid4().hex[:7].upper(),'user_id':user['u'],**data.model_dump(),'votes':0,'created_at':datetime.now(timezone.utc).isoformat()}
    insert_idea(item); return item

@app.post('/api/community/ideas/{idea_id}/vote')
def support_idea(idea_id:str,user=Depends(current_user)):
    if user['r'] not in ('CITIZEN','AUTHORITY','INSPECTOR'): raise HTTPException(403,'Forbidden')
    if vote_idea(idea_id)==0: raise HTTPException(404,'Solution not found.')
    return {'ok':True,'id':idea_id}

@app.post('/api/assign')
def assign(a:Assignment,user=Depends(require('AUTHORITY'))):
    detections=list_detections(1000)
    is_detection=any(d['id']==a.event_id for d in detections)
    is_complaint=any(c['id']==a.event_id for c in list_complaints())
    if not is_detection and not is_complaint: raise HTTPException(404,'Registered case not found.')
    target='DETECTION' if is_detection else 'CITIZEN_REPORT'
    anchor=next((d for d in detections if d['id']==a.event_id),None) if is_detection else None
    case_key=(f"VIDEO:{anchor.get('video_id')}" if anchor and anchor.get('video_id') else (a.case_id or a.event_id))
    existing=[x for x in list_assignments() if x.get('case_key')==case_key and x['status'] not in {'RESOLVED','REJECTED'}]
    if existing:
        return {**existing[0],'regional_officer':'Regional Road Officer'}
    x={'id':'A-'+uuid.uuid4().hex[:8].upper(),'event_id':a.event_id,'inspector_id':a.inspector_id,'status':'ASSIGNED','remarks':'Assigned by Traffic Authority','updated_at':datetime.now(timezone.utc).isoformat(),'target_type':target,'case_key':case_key}
    insert_assignment(x)
    if is_detection:
        # Assignment is case-level for a video: all physical tracks from that upload
        # receive the same operational lifecycle.
        for d in detections:
            if anchor and d.get('video_id')==anchor.get('video_id'):
                update_detection_status(d['id'],'ASSIGNED')
            elif not anchor.get('video_id') and d['id']==a.event_id:
                update_detection_status(d['id'],'ASSIGNED')
    else: update_complaint_status(a.event_id,'ASSIGNED')
    return {**x,'regional_officer':'Regional Road Officer'}

@app.get('/api/assignments')
def assignments(user=Depends(current_user)):
    if user['r']=='AUTHORITY': return [{**x,'regional_officer':'Regional Road Officer'} for x in list_assignments()]
    if user['r']=='INSPECTOR': return [{**x,'regional_officer':'Regional Road Officer'} for x in list_assignments(user['u'])]
    raise HTTPException(403,'Forbidden')

def sync_related_detection_status(event_id: str, status: str):
    """For a clustered pothole case, keep all close observations at the same lifecycle stage."""
    import sqlite3, math
    rows=list_detections(500)
    target=next((d for d in rows if d['id']==event_id), None)
    if not target:
        return
    for d in rows:
        if d['road'] != target['road']:
            continue
        # Within one uploaded video, track IDs represent distinct physical
        # potholes. Never propagate the lifecycle of one track to another.
        # A video upload is one operational case even when it contains multiple physical tracks.
        # Lifecycle updates therefore propagate to every track from that video.
        if target.get('video_id') and d.get('video_id') == target.get('video_id'):
            update_detection_status(d['id'], status)
            continue
        # Approximate distance in metres; good enough for the prototype cluster radius.
        dy=(d['lat']-target['lat'])*111_000
        dx=(d['lng']-target['lng'])*111_000*math.cos(math.radians(target['lat']))
        distance=(dx*dx+dy*dy)**0.5
        try:
            from datetime import datetime
            t1=datetime.fromisoformat(str(target['timestamp']).replace('Z','+00:00'))
            t2=datetime.fromisoformat(str(d['timestamp']).replace('Z','+00:00'))
            time_ok=abs((t1-t2).total_seconds()) <= 45*60
        except Exception:
            time_ok=True
        if distance <= 180 and time_ok:
            update_detection_status(d['id'], status)

@app.post('/api/assignments/verify')
def verify(v:Verify,user=Depends(require('INSPECTOR','AUTHORITY'))):
    status=v.status.upper()
    if status not in {'VERIFIED','REJECTED','WORK_INITIATED','RESOLVED'}: raise HTTPException(400,'Invalid workflow status.')
    rows=list_assignments(user['u'] if user['r']=='INSPECTOR' else None)
    matches=[a for a in rows if a['event_id']==v.event_id]
    if user['r']=='INSPECTOR' and not matches: raise HTTPException(403,'This case is not assigned to you.')
    if not matches:
        if not any(d['id']==v.event_id for d in list_detections(500)) and not any(c['id']==v.event_id for c in list_complaints()):
            raise HTTPException(404,'Registered case not found.')
    target=matches[0]['target_type'] if matches else ('DETECTION' if any(d['id']==v.event_id for d in list_detections(500)) else 'CITIZEN_REPORT')
    if target=='CITIZEN_REPORT':
        update_complaint_status(v.event_id,status)
    else:
        sync_related_detection_status(v.event_id,status)
    if matches:
        # Update the assignment row itself so both Authority and Officer see the same state.
        for a in matches:
            insert_assignment({'id':a['id'],'event_id':a['event_id'],'inspector_id':a['inspector_id'],'status':status,'remarks':v.remarks,'updated_at':datetime.now(timezone.utc).isoformat(),'target_type':target})
    return {'ok':True,'event_id':v.event_id,'status':status,'target_type':target}

@app.get('/api/reports/video/{video_id}')
def video_report(video_id:str,user=Depends(require('AUTHORITY','INSPECTOR'))):
    rows=[d for d in list_detections(1000) if d.get('video_id')==video_id]
    if not rows: raise HTTPException(404,'Video damage record not found.')
    run=get_video_run(video_id) or {}
    stored=get_damage_report(video_id)
    # The ingest path creates a deterministic report immediately. Never require
    # an Authority bearer token on the bus console just to draft the report.
    if stored:
        return {'video_id':video_id,'report':stored['report_text'],'facts':stored.get('facts',run),'detections':rows,'delivery':stored}
    text,facts=generate_fast_video_damage_report(run,rows)
    return {'video_id':video_id,'report':text,'facts':facts,'detections':rows,'delivery':{'status':'DRAFT'}}

@app.get('/api/reports')
def reports(user=Depends(require('AUTHORITY'))):
    return list_damage_reports(100)

@app.post('/api/reports/generate')
def report(user=Depends(require('AUTHORITY'))):
    data={'detections':list_detections(200),'fleet':list_telemetry(),'community':list_complaints(),'assignments':list_assignments()}
    return {'report':generate_report(data),'generated_at':datetime.now(timezone.utc).isoformat()}
