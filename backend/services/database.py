import sqlite3, json, os
from datetime import datetime, timezone

DB_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'urban.db'))

def conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c=sqlite3.connect(DB_PATH)
    c.row_factory=sqlite3.Row
    return c

def init_db():
    c=conn()
    c.executescript('''
    CREATE TABLE IF NOT EXISTS detections(
      id TEXT PRIMARY KEY,event_type TEXT,road TEXT,bus TEXT,camera TEXT,confidence REAL,
      lat REAL,lng REAL,severity TEXT,time TEXT,timestamp TEXT,bbox TEXT,evidence_url TEXT,source TEXT,status TEXT DEFAULT 'DETECTED'
    );
    CREATE TABLE IF NOT EXISTS telemetry(
      bus_id TEXT PRIMARY KEY,lat REAL,lng REAL,speed_kmh REAL,route TEXT,status TEXT,updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS complaints(
      id TEXT PRIMARY KEY,user_id TEXT,issue_type TEXT,location TEXT,description TEXT,lat REAL,lng REAL,status TEXT,created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS assignments(
      id TEXT PRIMARY KEY,event_id TEXT,inspector_id TEXT,status TEXT,remarks TEXT,updated_at TEXT,target_type TEXT DEFAULT 'DETECTION'
    );
    CREATE TABLE IF NOT EXISTS video_runs(
      video_id TEXT PRIMARY KEY,duration_sec REAL,processed_frames INTEGER,frame_stride INTEGER,raw_detections INTEGER,
      zone_detection_count INTEGER,zone_timestamp_sec REAL,processing_ms REAL,inference_ms REAL,created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS damage_reports(
      id TEXT PRIMARY KEY,video_id TEXT,report_type TEXT,report_text TEXT,facts_json TEXT,status TEXT DEFAULT 'SENT_TO_AUTHORITY',
      generated_at TEXT,recipient_role TEXT DEFAULT 'AUTHORITY',source_bus TEXT,road TEXT
    );
    CREATE TABLE IF NOT EXISTS community_ideas(
      id TEXT PRIMARY KEY,user_id TEXT,title TEXT,category TEXT,description TEXT,votes INTEGER DEFAULT 0,created_at TEXT,complaint_id TEXT
    );
    CREATE TABLE IF NOT EXISTS complaint_evidence(
      complaint_id TEXT PRIMARY KEY,evidence_url TEXT
    );
    ''')
    # Safe migration for databases created by the earlier prototype.
    cols={r['name'] for r in c.execute('PRAGMA table_info(detections)').fetchall()}
    if 'status' not in cols: c.execute("ALTER TABLE detections ADD COLUMN status TEXT DEFAULT 'DETECTED'")
    cols={r['name'] for r in c.execute('PRAGMA table_info(detections)').fetchall()}
    if 'video_id' not in cols: c.execute("ALTER TABLE detections ADD COLUMN video_id TEXT")
    if 'track_id' not in cols: c.execute("ALTER TABLE detections ADD COLUMN track_id TEXT")
    if 'frame_index' not in cols: c.execute("ALTER TABLE detections ADD COLUMN frame_index INTEGER")
    if 'video_timestamp_sec' not in cols: c.execute("ALTER TABLE detections ADD COLUMN video_timestamp_sec REAL")
    if 'bbox_width_px' not in cols: c.execute("ALTER TABLE detections ADD COLUMN bbox_width_px REAL")
    if 'bbox_height_px' not in cols: c.execute("ALTER TABLE detections ADD COLUMN bbox_height_px REAL")
    if 'bbox_area_px' not in cols: c.execute("ALTER TABLE detections ADD COLUMN bbox_area_px REAL")
    if 'bbox_width_ratio' not in cols: c.execute("ALTER TABLE detections ADD COLUMN bbox_width_ratio REAL")
    if 'bbox_height_ratio' not in cols: c.execute("ALTER TABLE detections ADD COLUMN bbox_height_ratio REAL")
    if 'bbox_area_ratio' not in cols: c.execute("ALTER TABLE detections ADD COLUMN bbox_area_ratio REAL")
    if 'track_hits' not in cols: c.execute("ALTER TABLE detections ADD COLUMN track_hits INTEGER")
    if 'evidence_score' not in cols: c.execute("ALTER TABLE detections ADD COLUMN evidence_score REAL")
    if 'context_evidence_url' not in cols: c.execute("ALTER TABLE detections ADD COLUMN context_evidence_url TEXT")
    if 'zone_evidence_url' not in cols: c.execute("ALTER TABLE detections ADD COLUMN zone_evidence_url TEXT")
    if 'context_frame_index' not in cols: c.execute("ALTER TABLE detections ADD COLUMN context_frame_index INTEGER")
    if 'context_timestamp_sec' not in cols: c.execute("ALTER TABLE detections ADD COLUMN context_timestamp_sec REAL")
    cols={r['name'] for r in c.execute('PRAGMA table_info(complaints)').fetchall()}
    if 'user_id' not in cols: c.execute("ALTER TABLE complaints ADD COLUMN user_id TEXT DEFAULT 'legacy-citizen'")
    cols={r['name'] for r in c.execute('PRAGMA table_info(assignments)').fetchall()}
    if 'target_type' not in cols: c.execute("ALTER TABLE assignments ADD COLUMN target_type TEXT DEFAULT 'DETECTION'")
    cols={r['name'] for r in c.execute('PRAGMA table_info(assignments)').fetchall()}
    if 'case_key' not in cols: c.execute("ALTER TABLE assignments ADD COLUMN case_key TEXT")
    cols={r['name'] for r in c.execute('PRAGMA table_info(community_ideas)').fetchall()}
    if 'complaint_id' not in cols: c.execute("ALTER TABLE community_ideas ADD COLUMN complaint_id TEXT")
    c.commit(); c.close()

def insert_detection(e):
    c=conn()
    c.execute("""INSERT OR REPLACE INTO detections
      (id,event_type,road,bus,camera,confidence,lat,lng,severity,time,timestamp,bbox,evidence_url,source,status,video_id,track_id,frame_index,video_timestamp_sec,bbox_width_px,bbox_height_px,bbox_area_px,bbox_width_ratio,bbox_height_ratio,bbox_area_ratio,track_hits,evidence_score,context_evidence_url,zone_evidence_url,context_frame_index,context_timestamp_sec)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
      (e['id'],e['event_type'],e['road'],e['bus'],e['camera'],e['confidence'],e['lat'],e['lng'],e['severity'],e['time'],e['timestamp'],json.dumps(e['bbox']),e['evidence_url'],e['source'],e.get('status','DETECTED'),e.get('video_id'),e.get('track_id'),e.get('frame_index'),e.get('video_timestamp_sec'),e.get('bbox_width_px'),e.get('bbox_height_px'),e.get('bbox_area_px'),e.get('bbox_width_ratio'),e.get('bbox_height_ratio'),e.get('bbox_area_ratio'),e.get('track_hits'),e.get('evidence_score'),e.get('context_evidence_url'),e.get('zone_evidence_url'),e.get('context_frame_index'),e.get('context_timestamp_sec')))
    c.commit(); c.close()

def list_detections(limit=100, status=None):
    c=conn()
    if status:
        rows=c.execute("SELECT * FROM detections WHERE status=? ORDER BY timestamp DESC LIMIT ?",(status,limit)).fetchall()
    else:
        rows=c.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",(limit,)).fetchall()
    c.close(); out=[]
    for r in rows:
        d=dict(r); d['bbox']=json.loads(d['bbox']); out.append(d)
    return out

def update_detection_status(event_id,status):
    c=conn(); c.execute("UPDATE detections SET status=? WHERE id=?",(status,event_id)); c.commit(); c.close()

def update_complaint_status(complaint_id,status):
    c=conn(); c.execute("UPDATE complaints SET status=? WHERE id=?",(status,complaint_id)); c.commit(); c.close()

def insert_telemetry(t):
    c=conn(); c.execute("INSERT OR REPLACE INTO telemetry VALUES(?,?,?,?,?,?,?)",
      (t['bus_id'],t['lat'],t['lng'],t.get('speed_kmh',0),t.get('route','Unknown'),t.get('status','ACTIVE'),datetime.now(timezone.utc).isoformat()))
    c.commit(); c.close()

def list_telemetry():
    c=conn(); rows=c.execute("SELECT * FROM telemetry ORDER BY bus_id").fetchall(); c.close(); return [dict(r) for r in rows]

def insert_complaint(x):
    c=conn(); c.execute("INSERT INTO complaints VALUES(?,?,?,?,?,?,?,?,?)",
      (x['id'],x.get('user_id','citizen'),x['issue_type'],x['location'],x.get('description',''),x.get('lat'),x.get('lng'),x['status'],x['created_at']))
    c.commit(); c.close()

def list_complaints(user_id=None):
    c=conn()
    if user_id:
        rows=c.execute("SELECT * FROM complaints WHERE user_id=? ORDER BY created_at DESC",(user_id,)).fetchall()
    else:
        rows=c.execute("SELECT * FROM complaints ORDER BY created_at DESC").fetchall()
    c.close(); return [dict(r) for r in rows]

def insert_assignment(a):
    c=conn(); c.execute("INSERT OR REPLACE INTO assignments(id,event_id,inspector_id,status,remarks,updated_at,target_type,case_key) VALUES(?,?,?,?,?,?,?,?)",
      (a['id'],a['event_id'],a['inspector_id'],a['status'],a.get('remarks',''),a['updated_at'],a.get('target_type','DETECTION'),a.get('case_key')))
    c.commit(); c.close()

def list_assigned_detections(inspector_id):
    c=conn()
    rows=c.execute("""SELECT d.* FROM detections d JOIN assignments a ON a.event_id=d.id WHERE a.inspector_id=? ORDER BY d.timestamp DESC""",(inspector_id,)).fetchall()
    c.close(); out=[]
    for r in rows:
        d=dict(r); d['bbox']=json.loads(d['bbox']); out.append(d)
    return out

def list_assignments(inspector_id=None):
    c=conn()
    where=''
    params=()
    if inspector_id:
        where='WHERE a.inspector_id=?'
        params=(inspector_id,)
    rows=c.execute(f"""SELECT a.*,
        COALESCE(d.road, c.location) AS road, COALESCE(d.bus, 'CITIZEN') AS bus, COALESCE(d.camera, 'CITIZEN REPORT') AS camera,
        COALESCE(d.confidence, 0) AS confidence, COALESCE(d.lat, c.lat) AS lat, COALESCE(d.lng, c.lng) AS lng,
        COALESCE(d.severity, CASE WHEN c.issue_type LIKE '%Pothole%' THEN 'HIGH' ELSE 'MEDIUM' END) AS severity,
        COALESCE(d.time, substr(c.created_at,12,8)) AS time, COALESCE(d.evidence_url, ce.evidence_url) AS evidence_url,
        COALESCE(d.status, c.status) AS event_status, COALESCE(d.event_type, c.issue_type) AS case_type,
        c.issue_type, c.description AS complaint_description
        FROM assignments a
        LEFT JOIN detections d ON a.event_id=d.id
        LEFT JOIN complaints c ON a.event_id=c.id
        LEFT JOIN complaint_evidence ce ON a.event_id=ce.complaint_id
        {where} ORDER BY a.updated_at DESC""",params).fetchall()
    c.close(); return [dict(r) for r in rows]

def tick_simulated_telemetry():
    import math, time
    c=conn(); rows=c.execute("SELECT * FROM telemetry").fetchall(); now=time.time()
    for i,r in enumerate(rows):
        lat=r['lat'] + math.sin(now/18+i)*0.00008
        lng=r['lng'] + math.cos(now/20+i)*0.00008
        c.execute("UPDATE telemetry SET lat=?,lng=?,updated_at=? WHERE bus_id=?",
                  (lat,lng,datetime.now(timezone.utc).isoformat(),r['bus_id']))
    c.commit(); c.close()


def insert_damage_report(x):
    c=conn(); c.execute("INSERT OR REPLACE INTO damage_reports(id,video_id,report_type,report_text,facts_json,status,generated_at,recipient_role,source_bus,road) VALUES(?,?,?,?,?,?,?,?,?,?)",
      (x['id'],x.get('video_id'),x.get('report_type','VIDEO_DAMAGE'),x['report_text'],json.dumps(x.get('facts',{}),default=str),x.get('status','SENT_TO_AUTHORITY'),x.get('generated_at') or datetime.now(timezone.utc).isoformat(),x.get('recipient_role','AUTHORITY'),x.get('source_bus'),x.get('road')))
    c.commit(); c.close()

def list_damage_reports(limit=50):
    c=conn(); rows=c.execute("SELECT * FROM damage_reports ORDER BY generated_at DESC LIMIT ?",(limit,)).fetchall(); c.close()
    out=[]
    for r in rows:
        d=dict(r)
        try: d['facts']=json.loads(d.get('facts_json') or '{}')
        except Exception: d['facts']={}
        d.pop('facts_json',None)
        out.append(d)
    return out

def get_damage_report(video_id):
    c=conn(); r=c.execute("SELECT * FROM damage_reports WHERE video_id=? ORDER BY generated_at DESC LIMIT 1",(video_id,)).fetchone(); c.close()
    if not r: return None
    d=dict(r)
    try: d['facts']=json.loads(d.get('facts_json') or '{}')
    except Exception: d['facts']={}
    d.pop('facts_json',None)
    return d

def insert_idea(x):
    c=conn(); c.execute("INSERT INTO community_ideas(id,user_id,title,category,description,votes,created_at,complaint_id) VALUES(?,?,?,?,?,?,?,?)",(x['id'],x['user_id'],x['title'],x['category'],x['description'],x.get('votes',0),x['created_at'],x.get('complaint_id'))); c.commit(); c.close()

def list_ideas():
    c=conn(); rows=c.execute("SELECT * FROM community_ideas ORDER BY votes DESC, created_at DESC").fetchall(); c.close(); return [dict(r) for r in rows]

def vote_idea(idea_id):
    c=conn(); c.execute("UPDATE community_ideas SET votes=votes+1 WHERE id=?",(idea_id,)); changed=c.total_changes; c.commit(); c.close(); return changed


def insert_video_run(x):
    c=conn(); c.execute("INSERT OR REPLACE INTO video_runs(video_id,duration_sec,processed_frames,frame_stride,raw_detections,zone_detection_count,zone_timestamp_sec,processing_ms,inference_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",(x['video_id'],x.get('duration_sec',0),x.get('processed_frames',0),x.get('frame_stride',1),x.get('raw_detections',0),x.get('zone_detection_count',0),x.get('zone_timestamp_sec'),x.get('processing_ms',0),x.get('inference_ms',0),x.get('created_at',datetime.now(timezone.utc).isoformat()))); c.commit(); c.close()

def get_video_run(video_id):
    c=conn(); r=c.execute("SELECT * FROM video_runs WHERE video_id=?",(video_id,)).fetchone(); c.close(); return dict(r) if r else None
