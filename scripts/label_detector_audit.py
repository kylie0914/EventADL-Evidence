"""브라우저에서 detector audit 프레임을 사람이 라벨링하는 작은 도구.

추가 패키지 없이 Python 표준 라이브러리만 사용한다. 모든 저장은 임시 파일을
거쳐 원본 CSV를 교체하므로, 도중에 브라우저를 닫아도 완료한 라벨은 남는다.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd


LABEL_COLUMNS = [
    "gt_hand_visible",
    "gt_object_visible",
    "gt_interaction",
    "gt_active_object",
    "review_notes",
]


HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ADL Detector Audit</title>
<style>
body{margin:0;background:#111827;color:#f9fafb;font-family:system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:16px}
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}.bar{height:12px;background:#374151;border-radius:9px;flex:1;min-width:220px}.fill{height:100%;background:#22c55e;border-radius:9px}
.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(330px,1fr);gap:18px;margin-top:14px}.image{background:#000;border-radius:12px;display:flex;align-items:center;justify-content:center;min-height:62vh}.image img{max-width:100%;max-height:76vh;object-fit:contain}
.panel{background:#1f2937;padding:18px;border-radius:12px}.question{margin:0 0 18px}.question strong{display:block;margin-bottom:8px;font-size:17px}.choices{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px}.choices button,button.nav{padding:12px;border:1px solid #4b5563;border-radius:8px;background:#374151;color:white;font-size:15px;cursor:pointer}.choices button.on{background:#2563eb;border-color:#60a5fa}.choices button.no.on{background:#dc2626}.choices button.blank.on{background:#6b7280}
input,textarea{box-sizing:border-box;width:100%;padding:10px;border-radius:7px;border:1px solid #6b7280;background:#111827;color:white}textarea{height:70px}.navs{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px}button.next{background:#16a34a}.meta{color:#cbd5e1;font-size:14px}.saved{color:#86efac;min-width:90px}details{margin-top:13px;color:#cbd5e1}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}.help{font-size:13px;color:#9ca3af}
@media(max-width:900px){.grid{grid-template-columns:1fr}.image{min-height:auto}.image img{max-height:55vh}}
</style></head><body><main>
<div class="top"><b id="counter"></b><div class="bar"><div class="fill" id="fill"></div></div><span id="progress"></span><span class="saved" id="saved"></span></div>
<div class="grid"><div class="image"><img id="frame"></div><section class="panel">
<div class="meta" id="meta"></div>
<div id="binary"></div>
<div class="question"><strong>Active object 이름</strong><input id="active" placeholder="예: cup, knife (없거나 모르면 비워두기)"></div>
<div class="question"><strong>메모 (선택)</strong><textarea id="notes" placeholder="가려짐, 애매함, detector 오탐 이유 등"></textarea></div>
<div class="navs"><button class="nav" onclick="move(-1)">← 이전</button><button class="nav next" onclick="move(1)">저장 후 다음 →</button></div>
<p class="help">자동 저장됩니다. 단축키: ←/→ 이동, 1=손, 2=물체, 3=상호작용 (각 키를 누르면 예/아니오 전환)</p>
<details><summary>모델 예측 보기 — 정답을 먼저 판단한 뒤 확인</summary><pre id="prediction"></pre></details>
</section></div></main>
<script>
const fields=[['gt_hand_visible','손이 보이나요?'],['gt_object_visible','조작 가능한 물체가 보이나요?'],['gt_interaction','손이 물체를 만지거나 조작 중인가요?']];
let rows=[],idx=0,timer=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function init(){rows=await (await fetch('/api/rows')).json(); const first=rows.findIndex(r=>!complete(r)); idx=first<0?0:first; render();}
function complete(r){return [r.gt_hand_visible,r.gt_object_visible,r.gt_interaction].every(v=>v===0||v===1)}
function render(){const r=rows[idx]; document.querySelector('#frame').src='/image/'+encodeURIComponent(r.image_path); document.querySelector('#counter').textContent=`${idx+1} / ${rows.length}`; const done=rows.filter(complete).length; document.querySelector('#progress').textContent=`완료 ${done}장 (${Math.round(done/rows.length*100)}%)`; document.querySelector('#fill').style.width=(done/rows.length*100)+'%'; document.querySelector('#meta').textContent=`시간 ${Number(r.time_sec).toFixed(1)}초 · frame ${r.frame_index}`;
document.querySelector('#binary').innerHTML=fields.map(([f,q])=>`<div class="question"><strong>${q}</strong><div class="choices"><button class="${r[f]===1?'on':''}" onclick="setv('${f}',1)">예</button><button class="no ${r[f]===0?'on':''}" onclick="setv('${f}',0)">아니오</button><button class="blank ${r[f]!==0&&r[f]!==1?'on':''}" onclick="setv('${f}',null)">미정</button></div></div>`).join('');
document.querySelector('#active').value=r.gt_active_object??''; document.querySelector('#notes').value=r.review_notes??''; document.querySelector('#prediction').textContent=JSON.stringify({hand_score:r.hand_score,object_score:r.object_score,active_object_score:r.active_object_score,hand_boxes:r.hand_boxes,objects:r.objects},null,2); document.querySelector('#saved').textContent='';}
function syncInputs(){const r=rows[idx];r.gt_active_object=document.querySelector('#active').value.trim();r.review_notes=document.querySelector('#notes').value.trim()}
function setv(f,v){syncInputs();rows[idx][f]=v;render();save();}
async function save(){clearTimeout(timer);syncInputs();const at=idx,r=rows[at];const labels=Object.fromEntries([...fields.map(x=>[x[0],r[x[0]]]),['gt_active_object',r.gt_active_object],['review_notes',r.review_notes]]);const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:at,labels})});document.querySelector('#saved').textContent=res.ok?'✓ 저장됨':'저장 실패';return res.ok}
function delayedSave(){clearTimeout(timer);syncInputs();timer=setTimeout(save,250)}
async function move(d){await save();idx=Math.max(0,Math.min(rows.length-1,idx+d));render();}
document.addEventListener('input',e=>{if(e.target.id==='active'||e.target.id==='notes')delayedSave()}); document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1);if(['1','2','3'].includes(e.key)){const f=fields[Number(e.key)-1][0];setv(f,rows[idx][f]===1?0:1)}}); init();
</script></body></html>"""


class AuditStore:
    def __init__(self, csv_path: Path, project_root: Path):
        self.csv_path = csv_path.resolve()
        self.project_root = project_root.resolve()
        self.lock = threading.Lock()
        self.table = pd.read_csv(self.csv_path)
        for column in LABEL_COLUMNS:
            if column not in self.table:
                self.table[column] = ""
        # 빈 문자열뿐인 CSV 열은 pandas가 float로 추정한다. object dtype으로 바꿔야
        # active-object 이름과 메모 같은 문자열을 나중에 안전하게 저장할 수 있다.
        self.table["gt_active_object"] = self.table["gt_active_object"].astype(object)
        self.table["review_notes"] = self.table["review_notes"].astype(object)

    @staticmethod
    def clean(value):
        return None if pd.isna(value) else value

    def public_rows(self):
        return [
            {key: self.clean(value) for key, value in row.items()}
            for row in self.table.to_dict(orient="records")
        ]

    def image(self, raw_path: str) -> Path:
        candidate = Path(unquote(raw_path))
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        candidate = candidate.resolve()
        audit_root = self.csv_path.parent.resolve()
        if audit_root not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def save(self, index: int, labels: dict):
        if not 0 <= index < len(self.table):
            raise IndexError(index)
        with self.lock:
            for column in LABEL_COLUMNS:
                if column in labels:
                    value = labels[column]
                    if column.startswith("gt_") and column != "gt_active_object":
                        value = "" if value is None else int(value)
                    self.table.at[index, column] = value
            temp = self.csv_path.with_suffix(".csv.tmp")
            self.table.to_csv(temp, index=False)
            os.replace(temp, self.csv_path)


def make_handler(store: AuditStore):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, content_type="application/json"):
            payload = data if isinstance(data, bytes) else data.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.send(200, HTML, "text/html; charset=utf-8")
            if path == "/api/rows":
                return self.send(200, json.dumps(store.public_rows(), ensure_ascii=False, allow_nan=False))
            if path.startswith("/image/"):
                try:
                    image = store.image(path.removeprefix("/image/"))
                    return self.send(200, image.read_bytes(), "image/jpeg")
                except FileNotFoundError:
                    return self.send(404, '{"error":"image not found"}')
            return self.send(404, '{"error":"not found"}')

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                return self.send(404, '{"error":"not found"}')
            try:
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size))
                store.save(int(body["index"]), body["labels"])
                return self.send(200, '{"ok":true}')
            except (ValueError, KeyError, IndexError, json.JSONDecodeError) as error:
                return self.send(400, json.dumps({"error": str(error)}))

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", default="data/audit/P04_01/audit_labels.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    store = AuditStore(project_root / args.audit_csv, project_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}"
    print(f"라벨링 화면: {url}")
    print("종료: 터미널에서 Ctrl+C (입력 내용은 자동 저장됨)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n라벨링 서버를 종료합니다.")


if __name__ == "__main__":
    main()
