"""Original WAL visibility/backfill model and benign sequential SQLite fixture."""

import argparse
from bisect import bisect_right
from contextlib import ExitStack,closing
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="9a9446f3b088bf3b8215311fbbdcc60a46045bed"
COMMANDS=[["reader","old"],["commit",{"1":"A1"}],["reader","middle"],
          ["commit",{"1":"A2","2":"B2"}],["checkpoint"],["release","old"],
          ["checkpoint"],["release","middle"],["checkpoint"],["reset"]]


def integer(v,low,high):
    if type(v) is not int or not low<=v<=high:
        raise ValueError(f"exact integer in {low}..{high} required")
    return v


def updates(data):
    if not isinstance(data,dict) or not data or len(data)>64:
        raise ValueError("one to64original page updates required")
    result={}
    for key,value in data.items():
        if type(key) is not int:
            raise ValueError("exact integer page IDs required")
        integer(key,1,64)
        if not isinstance(value,str) or len(value)>64 or any(not 32<=ord(c)<=126 for c in value):
            raise ValueError("bounded printable-ASCII original page value required")
        result[key]=value
    return result


class Wal:
    def __init__(self,initial):
        self.db=updates(initial);self.frames=[];self.head=0;self.backfill=0;self.readers={};self.used=set()

    def commit(self,data):
        data=updates(data)
        if len(self.frames)!=self.head:
            raise ValueError("original uncommitted tail must be recovered first")
        if set(data)-set(self.db):
            raise ValueError("original fixed database cannot add page IDs")
        if len(self.frames)+len(data)>4096:
            raise ValueError("original frame cap4096exceeded")
        for i,(page,value) in enumerate(sorted(data.items())):
            self.frames.append({"page":page,"value":value,"commit_end":i==len(data)-1})
        self.head=len(self.frames)

    def begin(self,name):
        if (not isinstance(name,str) or not name or len(name)>32 or not name.isascii() or
                any(not(c.isalnum() or c in "-_") for c in name) or name in self.used or len(self.readers)>=16):
            raise ValueError("bounded nonreused original reader name required")
        self.readers[name]=self.head;self.used.add(name)

    def release(self,name):
        if not isinstance(name,str) or name not in self.readers:
            raise ValueError("unknown or released original reader")
        del self.readers[name]

    def read(self,page,reader=None):
        integer(page,1,64)
        if page not in self.db:
            raise ValueError("page absent from original fixed database")
        if reader is not None and (not isinstance(reader,str) or reader not in self.readers):
            raise ValueError("unknown original reader")
        end=self.head if reader is None else self.readers[reader]
        for frame in reversed(self.frames[:end]):
            if frame["page"]==page:
                return frame["value"]
        return self.db[page]

    def checkpoint(self):
        safe=min([self.head,*self.readers.values()])
        for frame in self.frames[self.backfill:safe]:
            self.db[frame["page"]]=frame["value"]
        self.backfill=safe
        return {"eligible_head":safe,"backfilled":self.backfill,"committed_frames":self.head,
                "fully_backfilled":self.backfill==self.head}

    def reset(self):
        if self.readers or self.backfill!=self.head or len(self.frames)!=self.head:
            raise ValueError("original conservative reset requires no readers,complete backfill and no pending tail")
        self.frames=[];self.head=0;self.backfill=0

    def append_uncommitted(self,page,value):
        data=updates({page:value})
        if page not in self.db or len(self.frames)>=4096:
            raise ValueError("original pending page/cap outside fixed model")
        self.frames.append({"page":page,"value":data[page],"commit_end":False})

    def recover(self):
        committed=max([0,*[i+1 for i,f in enumerate(self.frames) if f["commit_end"]]])
        self.frames=self.frames[:committed];self.head=committed

    def snapshot(self):
        return {"committed_frames":self.head,"model_frames":len(self.frames),"backfilled":self.backfill,
                "database":{str(k):v for k,v in sorted(self.db.items())},"readers":dict(sorted(self.readers.items())),
                "current":{str(k):self.read(k) for k in sorted(self.db)},
                "reader_values":{name:{str(k):self.read(k,name) for k in sorted(self.db)} for name in sorted(self.readers)}}


def play(commands):
    if not isinstance(commands,list) or len(commands)>128:
        raise ValueError("at most128original commands required")
    wal=Wal({1:"A0",2:"B0"});trace=[]
    for cmd in commands:
        if not isinstance(cmd,list) or not cmd:
            raise ValueError("nonempty command array required")
        op=cmd[0];result=None
        if op=="commit" and len(cmd)==2 and isinstance(cmd[1],dict):
            data={}
            for k,v in cmd[1].items():
                if type(k) is not str or k not in ("1","2"):
                    raise ValueError("workbench page keys must be1or2")
                data[int(k)]=v
            wal.commit(data)
        elif op=="reader" and len(cmd)==2:
            wal.begin(cmd[1])
        elif op=="release" and len(cmd)==2:
            wal.release(cmd[1])
        elif op=="checkpoint" and len(cmd)==1:
            result=wal.checkpoint()
        elif op=="reset" and len(cmd)==1:
            wal.reset()
        else:
            raise ValueError("unsupported original command or argument count")
        trace.append({"command":cmd,"result":result,"snapshot":wal.snapshot()})
    return trace


def budget(p):
    if not isinstance(p,dict) or set(p)!={"frames","page_bytes"}:
        raise ValueError("exact selected frame/page fields required")
    frames=integer(p["frames"],1,2**20);size=integer(p["page_bytes"],512,65536)
    if size&(size-1):
        raise ValueError("power-of-two selected page size required")
    return {"wal_header_bytes":32,"frame_header_bytes":24,"selected_frame_bytes":24+size,
            "selected_wal_bytes":32+frames*(24+size),
            "scope":"selected WAL header+frame layout only;not main DB,SHM,filesystem allocation,retained capacity or a binary writer"}


def sqlite_fixture():
    with tempfile.TemporaryDirectory(prefix="original-wal-",dir=HERE) as folder:
        with ExitStack() as stack:
            path=Path(folder)/"original.db"
            def connect():
                return stack.enter_context(closing(sqlite3.connect(path,isolation_level=None,timeout=0)))
            writer=connect();old=connect();middle=connect();probe=connect();late=connect()
            mode=writer.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode!="wal":
                raise RuntimeError(f"original fixture failed to activate WAL:{mode}")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE original(k INTEGER PRIMARY KEY,v TEXT)")
            writer.execute("INSERT INTO original VALUES(1,'A0'),(2,'B0')")
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            def rows(conn):
                return [v for (v,) in conn.execute("SELECT v FROM original ORDER BY k").fetchall()]
            late.execute("BEGIN")
            old.execute("BEGIN");before=rows(old)
            writer.execute("UPDATE original SET v='A1' WHERE k=1")
            middle.execute("BEGIN");middle_before=rows(middle)
            writer.execute("BEGIN");writer.execute("UPDATE original SET v='A2' WHERE k=1")
            writer.execute("UPDATE original SET v='B2' WHERE k=2");writer.execute("COMMIT")
            held_old=rows(old);held_middle=rows(middle);latest=rows(probe)
            delayed_begin=rows(late);late.execute("ROLLBACK")
            writer.execute("BEGIN");writer.execute("UPDATE original SET v='UNCOMMITTED' WHERE k=2")
            uncommitted_hidden=rows(probe);writer.execute("ROLLBACK")
            passive=list(writer.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone())
            held_truncate=list(writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
            old.execute("ROLLBACK");middle.execute("ROLLBACK")
            released_truncate=list(writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
            bytes_after=Path(str(path)+"-wal").stat().st_size
            return {"provenance":"measured","sqlite_version":sqlite3.sqlite_version,"mode":mode,
                    "schedule":"original sequential same-thread connections;no concurrent write/checkpoint race or corruption test",
                    "old_before":before,"middle_before":middle_before,"old_after_commits":held_old,
                    "middle_after_commits":held_middle,"latest":latest,"deferred_begin_first_read":delayed_begin,
                    "uncommitted_hidden":uncommitted_hidden,
                    "passive_held_reader":passive,"truncate_held_reader":held_truncate,
                    "truncate_after_release":released_truncate,"wal_bytes_after_successful_truncate":bytes_after,
                    "patch_status":"not audited;compare runtime version/backport status with documented WAL-reset fixed3.51.3or named backports;fixture is not production certification"}


def timing_paths(n):
    frames=[(i%32,(i*17)%65536) for i in range(n)]
    index={}
    for pos,(page,value) in enumerate(frames,1):
        index.setdefault(page,[]).append((pos,value))
    positions={page:[v[0] for v in records] for page,records in index.items()}
    queries=[(i%32,(i*40503+17)%n+1) for i in range(64)]
    def indexed():
        out=[]
        for page,end in queries:
            records=index.get(page,[]);at=bisect_right(positions.get(page,[]),end)-1
            out.append(records[at][1] if at>=0 else None)
        return out
    def scan():
        out=[]
        for page,end in queries:
            value=None
            for pos in range(end-1,-1,-1):
                if frames[pos][0]==page:
                    value=frames[pos][1];break
            out.append(value)
        return out
    return indexed,scan


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n);assert funcs[0]()==funcs[1]()
        for _ in range(2):
            for f in funcs:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    funcs[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"frames":n,"queries":64,"indexed_samples_ms":samples[0],"scan_samples_ms":samples[1],
                     "indexed_median_ms":statistics.median(samples[0]),"scan_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"64original snapshot page lookups via prebuilt per-page positions+bisect or reverse frame scan",
            "excludes":"index construction,Wal class,SQLite SQL engine/WAL-index,locks,files,fsync,checkpoint,durability","rows":rows}


def build_report(latency,native):
    wal=Wal({1:"A0",2:"B0"});wal.commit({1:"A1"});wal.commit({1:"A2",2:"B2"})
    before=wal.snapshot();wal.append_uncommitted(2,"UNCOMMITTED");pending=wal.snapshot()
    wal.recover();recovered=wal.snapshot()
    return {"schema_version":1,"round":142,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sqlite/sqlite","pin":PIN,"license":"SQLite public-domain source;no implementation mirrored",
                      "verified":"reader fixed commit end marks,latest page version at/before end mark,checkpoint versus commit,passive/full/restart/truncate distinction",
                      "chronology":"Oct23read-only-SHM recovery error race fix;not invention of WAL;WAL-reset fix documented3.51.3and named backports",
                      "scope":"original fixed-page sequential ledger;no native locks,checksum/salt recovery,fsync,crash persistence,WAL-index read slots or reader-refresh semantics"},
            "case":{"commands":COMMANDS,"trace":play(COMMANDS),"uncommitted_tail":{"before":before,"pending":pending,"recovered":recovered},
                    "selected_budget":{"inputs":{"frames":3,"page_bytes":4096},"outputs":budget({"frames":3,"page_bytes":4096})},
                    "reset_policy":"original conservative no-active-readers gate;native database-only readers/read-mark slots not fully simulated"},
            "native_sqlite_contract_fixture":native,"latency":latency,
            "native_checkpoint_latency_ms":None,"native_crash_durability_verified":None,"race_probability":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
        (HERE/"native_oracle.json").write_text(json.dumps(sqlite_fixture(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")),
                        json.loads((HERE/"native_oracle.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round142 sealed artifacts match" if a.check else "Wrote Round142 original artifacts")


if __name__=="__main__":
    main()
