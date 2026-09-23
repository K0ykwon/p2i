import {useEffect,useState} from 'react';
import {sampleIndex} from '../execution';
import type {Tensor} from '../types/model';
export interface Sample {elements:number;values:(number|null)[]|null;message:string;min?:number|null;max?:number|null;mean?:number|null;std?:number|null}
export interface Inspection {tensors:Record<string,Sample>;operations:Record<string,unknown>}
export const emptyInspection:Inspection={tensors:{},operations:{}};
export function TensorSample({tensor,sample}:{tensor:Tensor;sample?:Sample}){
 const [hover,setHover]=useState<number|null>(null);
 const [coordinates,setCoordinates]=useState<Record<number,number>>({});
 useEffect(()=>{setCoordinates({});setHover(null)},[tensor.id]);
 const values=sample?.values?.slice(0,64);
 if(!values?.length)return <div className="sample-omitted">Values unavailable · [{tensor.shape.join(', ')}]{sample?` · ${sample.elements.toLocaleString()} elements`:''}<small>{sample?.message??'Enable bounded capture in Edit, then Build & Re-trace.'}</small></div>;
 const complete=sample!.elements===values.length;
 const dimensions=tensor.shape;
 const concrete=dimensions.every(d=>typeof d==='number'&&Number.isInteger(d)&&d>=0);
 const dims=concrete?dimensions as number[]:[];
 const rank=dimensions.length;
 const matrix=concrete&&rank>=2;
 const cols=matrix?Math.max(1,Math.min(8,dims[rank-1])):Math.min(8,values.length);
 const rows=matrix?Math.min(8,dims[rank-2]):1;
 const cells=matrix?Array.from({length:rows*cols},(_,i)=>{
   const coords=dims.map((_,axis)=>coordinates[axis]??0);coords[rank-2]+=Math.floor(i/cols);coords[rank-1]+=i%cols;
   return sampleIndex(dims,coords,values.length);
 }):values.map((_,i)=>i);
 const scale=Math.max(1e-12,...values.filter((v):v is number=>v!==null).map(Math.abs));
 return <div className="tensor-sample"><div className="sample-caption"><strong>{rank===0?'Scalar':rank===1?'Vector strip':matrix?rank===2?'Matrix heatmap':'2D slice · last two axes':'Flat prefix (symbolic shape)'}</strong><span>{values.length} / {sample!.elements.toLocaleString()} values</span></div><p>Original shape [{dimensions.join(', ')}] · {complete?'complete capture':'sampled / truncated flat prefix'}. {matrix&&'At most 8 × 8 cells; uncaptured coordinates remain unavailable.'}</p>{matrix&&<div className="slice-controls">{dims.map((d,axis)=><label key={axis}>Axis {axis} {axis>=rank-2?'start':'index'}<input type="number" min="0" max={Math.max(0,d-1)} value={coordinates[axis]??0} onChange={e=>setCoordinates(c=>({...c,[axis]:Math.max(0,Math.min(d-1,Number(e.target.value)))}))}/></label>)}</div>}<div className="heatmap" role="group" aria-label={`Captured values for ${tensor.id}`} style={{gridTemplateColumns:`repeat(${cols},minmax(24px,1fr))`}}>{cells.map((index,cell)=>{const i=index??-1;const v=index===null?null:values[index];return <button key={cell} disabled={index===null} onMouseEnter={()=>setHover(i)} onMouseLeave={()=>setHover(null)} onFocus={()=>setHover(i)} onBlur={()=>setHover(null)} aria-label={`Value ${i}: ${v??'non-finite'}`} style={{background:v===null?'#eee':`rgba(${v<0?'191,111,58':'18,132,132'},${.08+.67*Math.abs(v)/scale})`}} title={`flat[${i}] = ${v??'non-finite'}`}>{index===null?'—':v===null?'?':Math.abs(v)<.01?v.toExponential(0):v.toFixed(1)}</button>})}</div><p className="sample-readout">{hover===null?'Hover or focus a cell to inspect its recorded value.':`flat[${hover}] = ${values[hover]??'non-finite'}`}</p><div className="sample-stats">{(['min','max','mean','std'] as const).map(k=><span key={k}>{k} <b>{sample![k]==null?'—':sample![k]!.toPrecision(3)}</b></span>)}</div><small>Teal positive · amber negative. Color scale ±{scale.toPrecision(3)} for this sample. Statistics describe the captured tensor.</small></div>;
}
