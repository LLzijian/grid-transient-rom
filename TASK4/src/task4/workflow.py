"""Small persistent research state machine, not an autonomous LLM/scheduler.
Decisions are made by the executing agent under AGENTS.md; this module enforces
recording, approval gates, budget accounting and accepted-candidate pointers.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json, os, re
from .paths import ROOT, project_path, read_json, write_json

def now():return datetime.now(timezone.utc).isoformat()

class Workflow:
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self.file=self.root/'state/research.json'
    def load(self):return read_json(self.file)
    @contextmanager
    def transaction(self):
        lock=self.root/'state/.state.lock'; lock.parent.mkdir(exist_ok=True)
        try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:raise RuntimeError('State lock exists; inspect active processes before manually recovering')
        try:
            os.write(fd,str(os.getpid()).encode());os.close(fd)
            s=self.load();yield s;s['updated_at']=now();write_json(self.file,s,self.root)
        finally:
            lock.unlink(missing_ok=True)
    def record(self,s,kind,**data):
        s['history'].append({'at':now(),'kind':kind,**data})
    def authorize(self,quote,cap):
        if not quote.strip() or cap<=0:raise ValueError('Actual user instruction and positive cap required')
        with self.transaction() as s:
            if s.get('pending_direction'):raise RuntimeError('Direction approval required before research can resume')
            if cap<s['budget']['spent_seconds']:raise ValueError('Cap smaller than recorded spent amount')
            s['authorization']={'user_instruction':quote,'recorded_at':now(),'source':'user_message_recorded_by_agent'}
            s['budget']['cap_seconds']=float(cap);s['status']='READY'
            self.record(s,'AUTHORIZED',cap_seconds=cap,quote=quote)
    def open_loop(self,loop_id,spec,reserve):
        if not re.fullmatch(r'T4_\d{4}',loop_id):raise ValueError('Loop IDs use T4_0001 format')
        spec=project_path(spec,self.root)
        if not spec.is_file():raise FileNotFoundError('Write the idea protocol first')
        with self.transaction() as s:
            if s['status'] not in ['READY','RUNNING'] or not s['authorization']:raise PermissionError('Explicit launch authorization required')
            if s.get('pending_direction'):raise PermissionError('WAITING_USER_DIRECTION_APPROVAL')
            if s['active_loop']:raise RuntimeError('Close active loop first')
            if loop_id in s['loops']:raise ValueError('Loop ID already used; do not reopen closed protocols')
            available=s['budget']['cap_seconds']-s['budget']['spent_seconds']
            if reserve<=0 or reserve>available+1e-6:raise ValueError(f'Reservation exceeds available {available:.2f}s')
            s['loops'][loop_id]={'spec':str(spec.relative_to(self.root)),'reserved_seconds':float(reserve),
                'spent_seconds':0.0,'status':'RUNNING','opened_at':now(),
                'accepted_before':s['accepted_candidate'],'accepted_code_before':s['accepted_code_ref']}
            s['active_loop']=loop_id;s['status']='RUNNING'
            self.record(s,'LOOP_OPENED',loop_id=loop_id,reserved_seconds=reserve)
    def process_started(self,loop_id,run_id,pid,command):
        with self.transaction() as s:
            if s['status']!='RUNNING' or s['active_loop']!=loop_id:raise PermissionError('No active authorized loop')
            if s['active_jobs']:raise RuntimeError('Default is one managed job; no nested runner billing')
            s['active_jobs'][run_id]={'pid':pid,'command':command,'started_at':now(),'loop_id':loop_id}
    def process_finished(self,run_id,seconds,exit_code,reason):
        with self.transaction() as s:
            job=s['active_jobs'].pop(run_id)
            amount=max(0.0,float(seconds));loop=s['loops'][job['loop_id']]
            loop['spent_seconds']+=amount;s['budget']['spent_seconds']+=amount
            self.record(s,'PROCESS_CLOSED',run_id=run_id,seconds=amount,exit_code=exit_code,reason=reason)
    def account(self,seconds,reason):
        if seconds<0 or not reason:raise ValueError('Explicit nonnegative cost and description required')
        with self.transaction() as s:
            if s['active_jobs']:raise RuntimeError('Avoid double counting time inside a managed process')
            if not s['active_loop']:raise RuntimeError('Open loop before accounting')
            s['loops'][s['active_loop']]['spent_seconds']+=seconds;s['budget']['spent_seconds']+=seconds
            self.record(s,'EXTERNAL_COST',seconds=seconds,reason=reason)
    def close_loop(self,decision,evidence,candidate=None,code_ref=None):
        if decision not in ['accept','retain','reject','inconclusive']:raise ValueError('Unknown decision')
        evidence=project_path(evidence,self.root)
        if not evidence.is_file():raise FileNotFoundError('A closure report is required')
        if candidate:
            candidate=project_path(candidate,self.root)
            if not candidate.exists():raise FileNotFoundError(candidate)
        if decision=='retain' and not candidate:raise ValueError('A real candidate path is required')
        if decision=='accept' and not code_ref:raise ValueError('Accepted source commit or snapshot reference required')
        with self.transaction() as s:
            if s['active_jobs']:raise RuntimeError('Wait for/checkpoint/terminate active jobs before closure')
            lid=s['active_loop']
            if not lid:raise ValueError('No active loop')
            loop=s['loops'][lid];loop.update(status='CLOSED',decision=decision,evidence=str(evidence.relative_to(self.root)),closed_at=now())
            loop['released_seconds']=max(0.0,loop['reserved_seconds']-loop['spent_seconds'])
            # Reservation is NOT actual spend: all unused allowance returns automatically.
            if decision=='accept':
                
                if candidate:s['accepted_candidate']=str(candidate.relative_to(self.root))
                s['accepted_code_ref']=code_ref
                s['active_candidate']=s['accepted_candidate']
            elif decision=='retain':
                cp=str(candidate.relative_to(self.root))
                if cp not in s['complementary_candidates']:s['complementary_candidates'].append(cp)
                s['active_candidate']=s['accepted_candidate']
            else:
                # Roll back pointers, never delete negative evidence or overwrite checkpoints.
                s['active_candidate']=s['accepted_candidate']
            s['active_loop']=None;s['status']='READY'
            if s['budget']['spent_seconds']>=s['budget']['cap_seconds']:s['status']='BLOCKED_RESOURCE'
            self.record(s,'LOOP_CLOSED',loop_id=lid,decision=decision,released_seconds=loop['released_seconds'])
    def propose_direction(self,proposal):
        proposal=project_path(proposal,self.root)
        if not proposal.is_file():raise FileNotFoundError(proposal)
        with self.transaction() as s:
            if s['active_jobs'] or s['active_loop']:raise RuntimeError('Close or checkpoint current experiment first; do not start new-direction jobs')
            if s.get('pending_direction'):raise RuntimeError('An approval request is already pending')
            s['pending_direction']={'proposal':str(proposal.relative_to(self.root)),'at':now(),'revision':s['route_revision']+1}
            s['status']='WAITING_USER_DIRECTION_APPROVAL'
            self.record(s,'DIRECTION_PROPOSED',proposal=str(proposal.relative_to(self.root)))
    def approve_direction(self,quote):
        if not quote.strip():raise ValueError('Record the actual explicit approval message, not inferred consent')
        with self.transaction() as s:
            if s['status']!='WAITING_USER_DIRECTION_APPROVAL':raise RuntimeError('No pending direction')
            p=s['pending_direction'];s['route_revision']=p['revision'];s['pending_direction']=None
            s['status']='READY' if s['authorization'] else 'NOT_STARTED'
            self.record(s,'USER_APPROVED_DIRECTION',proposal=p['proposal'],quote=quote)
    def reject_direction(self,quote):
        with self.transaction() as s:
            p=s.get('pending_direction')
            if not p or not quote.strip():raise ValueError('Pending proposal and explicit reply required')
            s['pending_direction']=None;s['status']='PAUSED'
            self.record(s,'USER_REJECTED_DIRECTION',proposal=p['proposal'],quote=quote)
    def pause(self,reason):
        with self.transaction() as s:
            if s['active_jobs']:raise RuntimeError('Terminate/checkpoint managed jobs before recording pause')
            s['status']='PAUSED';self.record(s,'PAUSE',reason=reason)
