import PipelineView from '../components/PipelineView'

export default function Home() {
  return (
    <div style={{fontFamily: 'Inter, Arial, sans-serif', padding: 24}}>
      <h1>FinAlze — Pipeline Overview</h1>
      <p>Pipeline stages and real-time status (POC).</p>
      <PipelineView />
    </div>
  )
}
