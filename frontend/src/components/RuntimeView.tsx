import {useEffect,useState} from 'react';
import type {IR,Selection} from '../types/model';
import {shape} from '../types/model';
import {GraphView} from './GraphView';
import {emptyInspection} from './TensorSample';
import type {Inspection} from './TensorSample';
export function RuntimeView({ir,scope,subtree,select,inspection=emptyInspection,onActiveModule}:{ir:IR;scope:string;subtree:Set<string>;select:(s:Selection)=>void;inspection?:Inspection;onActiveModule?:(id:string)=>void}){
 const calls=(ir.runtime?.calls??[]).filter(c=>subtree.has(c.module_id));const [step,setStep]=useState(0),[playing,setPlaying]=useState(false);
 useEffect(()=>{setStep(0);setPlaying(false)},[scope,ir]);
 useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setStep(s=>{if(s>=calls.length-1){setPlaying(false);return s}return s+1}),1000);return()=>clearInterval(timer)},[playing,calls.length]);
 const current=calls[step];
 useEffect(()=>{onActiveModule?.(current?.module_id??'')},[current?.module_id,onActiveModule]);
 const [operation,setOperation]=useState<Selection>({kind:'module',id:''});
 const currentOps=ir.operations.filter(o=>o.graph==='runtime'&&o.call_id===current?.id);
const modules=new Map(ir.modules.map(m=>[m.id,m]));const tensors=new Map(ir.tensors.map(t=>[t.id,t]));
 return <section className="runtime-view"><div className="runtime-controls"><button disabled={!step} onClick={()=>setStep(s=>s-1)}>Previous</button><button disabled={!calls.length} onClick={()=>setPlaying(p=>!p)}>{playing?'Pause':'Play'}</button><button disabled={step>=calls.length-1} onClick={()=>setStep(s=>s+1)}>Next</button><strong>Call {calls.length?step+1:0} / {calls.length}</strong></div><p className="hint">Recorded module entry order, including nested calls. This is not a timing or concurrency profiler.</p>{current&&<div className="current-call"><span>RECORDED CALL · {current.id}</span><h2>{modules.get(current.module_id)?.qualified_name||modules.get(current.module_id)?.name}</h2><p>{current.completed?'Forward completed':'Incomplete forward'} · parent call {current.parent_call_id??'none'}</p><div className="runtime-tensors">{current.input_tensor_ids.map(id=><button key={id} onClick={()=>select({kind:'tensor',id})}>{shape(tensors.get(id))}</button>)}<span>→</span>{current.output_tensor_ids.map(id=><button key={id} onClick={()=>select({kind:'tensor',id})}>{shape(tensors.get(id))}</button>)}</div></div>}<div className="runtime-computation"><h3>Operations recorded inside this call</h3>{currentOps.length?<GraphView key={current?.id} ir={ir} ops={currentOps} selected={operation} select={s=>{setOperation(s);select(s)}} inspection={inspection}/>:<p className="hint">No primitive operations are attached directly to this call. Step into the recorded child calls to inspect their computation.</p>}</div><details className="call-history"><summary>Recorded call list · {calls.length} calls</summary><div className="call-list">{calls.slice(Math.max(0,step-15),step+35).map(c=><button key={c.id} className={current?.id===c.id?'current':''} onClick={()=>setStep(calls.indexOf(c))}><code>#{c.sequence} {c.id}</code><span>{modules.get(c.module_id)?.qualified_name||modules.get(c.module_id)?.name}</span><small>{c.completed?'completed':'incomplete'}</small></button>)}</div></details>{!calls.length&&<p>No module calls were recorded in this subtree.</p>}</section>
}
