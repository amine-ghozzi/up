import {useEffect, useState} from 'react'

const stepsDefault = [
  {id: 'classify', name: 'Classification Gate', status: 'pending'},
  {id: 'tier0', name: 'Tier 0 — Native PDF', status: 'pending'},
  {id: 'tier1', name: 'Tier 1 — Fast OCR', status: 'pending'},
  {id: 'tier2', name: 'Tier 2 — VLM', status: 'pending'},
  {id: 'tier3', name: 'Tier 3 — Ensemble/HITL', status: 'pending'},
]

function Badge({status}){
  const color = status === 'passed' ? '#16a34a' : status === 'failed' ? '#dc2626' : '#f59e0b'
  return <span style={{background:color,color:'#fff',padding:'4px 8px',borderRadius:6,fontSize:12}}>{status}</span>
}

export default function PipelineView(){
  const [steps, setSteps] = useState(stepsDefault)

  // POC: simulate progress
  useEffect(()=>{
    let i = 0
    const t = setInterval(()=>{
      setSteps(prev=>{
        const copy = prev.map((s,idx)=>{
          if(idx < i) return {...s, status: 'passed'}
          if(idx === i) return {...s, status: 'running'}
          return s
        })
        return copy
      })
      i++
      if(i>stepsDefault.length) clearInterval(t)
    }, 800)
    return ()=>clearInterval(t)
  },[])

  return (
    <div style={{display:'flex',gap:20,alignItems:'flex-start',flexWrap:'wrap'}}>
      {steps.map((s,idx)=> (
        <div key={s.id} style={{border:'1px solid #e5e7eb',padding:16,borderRadius:8,width:260,boxShadow:'0 1px 2px rgba(0,0,0,0.04)'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
            <strong>{s.name}</strong>
            <Badge status={s.status} />
          </div>
          <div style={{marginTop:8,fontSize:13,color:'#374151'}}>
            <p style={{margin:0}}>QCS: <strong>{s.qcs ?? '—'}</strong></p>
            <p style={{margin:0}}>Confidence: <strong>{s.confidence ?? '—'}</strong></p>
          </div>
          <div style={{marginTop:12}}>
            <details>
              <summary style={{cursor:'pointer'}}>Details</summary>
              <pre style={{whiteSpace:'pre-wrap',fontSize:12}}>Placeholder logs for {s.name}</pre>
            </details>
          </div>
        </div>
      ))}
      <div style={{flexBasis:'100%'}}>
        <h3>Mirror Document / Data</h3>
        <div style={{border:'1px dashed #e5e7eb',padding:12,borderRadius:6}}>Upload UI & mirrored data will appear here in the full implementation.</div>
      </div>
    </div>
  )
}
