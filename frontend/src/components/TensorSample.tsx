import {useState} from 'react';
import type {Tensor} from '../types/model';
export interface Sample {elements:number;values:(number|null)[]|null;message:string;min?:number|null;max?:number|null;mean?:number|null;std?:number|null}
export interface Inspection {tensors:Record<string,Sample>;operations:Record<string,unknown>}
export const emptyInspection:Inspection={tensors:{},operations:{}};
export function TensorSample({tensor,sample}:{tensor:Tensor;sample?:Sample}){
 const [hover,setHover]=useState<number|null>(null);
 const values=sample?.values?.slice(0,64);
 if(!values?.length)return <div className="sample-omitted">Values not captured{sample?` · ${sample.elements.toLocaleString()} elements`:''}<small>{sample?.message??'Enable bounded capture in Edit, then Build & Re-trace.'}</small></div>;
 const complete=sample!.elements===values.length;
 const matrix=complete&&tensor.shape.length===2&&typeof tensor.shape[1]==='number'&&tensor.shape[1]<=16;
 const cols=matrix?tensor.shape[1] as number:Math.min(8,values.length);
 const scale=Math.max(1e-12,...values.filter((v):v is number=>v!==null).map(Math.abs));
 return <div className="tensor-sample"><div className="sample-caption"><strong>{matrix?'Complete matrix':complete?'All values · flat order':'Flat prefix sample'}</strong><span>{values.length} / {sample!.elements.toLocaleString()} values</span></div><div className="heatmap" role="group" aria-label={`Captured values for ${tensor.id}`} style={{gridTemplateColumns:`repeat(${cols},minmax(24px,1fr))`}}>{values.map((v,i)=><button key={i} onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)} onFocus={()=>setHover(i)} onBlur={()=>setHover(null)} aria-label={`Value ${i}: ${v??'non-finite'}`} style={{background:v===null?'#eee':`rgba(${v<0?'191,111,58':'18,132,132'},${.08+.67*Math.abs(v)/scale})`}} title={`flat[${i}] = ${v??'non-finite'}`}>{v===null?'?':Math.abs(v)<.01?v.toExponential(0):v.toFixed(1)}</button>)}</div><p className="sample-readout">{hover===null?'Hover or focus a cell to inspect its recorded value.':`flat[${hover}] = ${values[hover]??'non-finite'}`}</p><div className="sample-stats">{(['min','max','mean','std'] as const).map(k=><span key={k}>{k} <b>{sample![k]==null?'—':sample![k]!.toPrecision(3)}</b></span>)}</div><small>Teal positive · amber negative. Color scale ±{scale.toPrecision(3)} for this sample. Statistics describe the captured tensor.</small></div>;
}
