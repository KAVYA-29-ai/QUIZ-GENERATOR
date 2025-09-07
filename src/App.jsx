import React, {useState, useRef, useEffect} from 'react'
import { motion } from 'framer-motion'
import { jsPDF } from 'jspdf'

const DEFAULT_COUNT = 10

function humanFileToBase64(file){
  return new Promise((res, rej)=>{
    const r = new FileReader()
    r.onload = ()=>res(r.result.split(',')[1])
    r.onerror = rej
    r.readAsDataURL(file)
  })
}

export default function App(){
  const [loading, setLoading] = useState(false)
  const [textInput, setTextInput] = useState('')
  const [questions, setQuestions] = useState([])
  const [count, setCount] = useState(DEFAULT_COUNT)
  const [type, setType] = useState('mixed')
  const [topic, setTopic] = useState('')
  const [sourceType, setSourceType] = useState('text')
  const fileRef = useRef()

  useEffect(()=>{ // load from localStorage
    const s = localStorage.getItem('qq_questions')
    if (s) setQuestions(JSON.parse(s))
  },[])
  useEffect(()=> localStorage.setItem('qq_questions', JSON.stringify(questions)),[questions])

  async function handleFile(e){
    const f = e.target.files[0]
    if (!f) return
    const ext = f.name.split('.').pop().toLowerCase()
    if (ext==='pdf'){
      const base64 = await humanFileToBase64(f)
      setTextInput('')
      setSourceType('pdf')
      await generateFromServer({source:'pdf', content: base64})
    } else {
      const txt = await f.text()
      setTextInput(txt)
      setSourceType('text')
    }
  }

  async function generateFromServer(payloadExtra){
    setLoading(true)
    try{
      const body = {
        source: payloadExtra?.source || sourceType,
        content: payloadExtra?.content || textInput,
        options: { count: Number(count||DEFAULT_COUNT), type, topic }
      }
      const res = await fetch('/.netlify/functions/generate', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json?.error||'Server error')
      if (json.questions) setQuestions(json.questions)
      else setQuestions([])
    }catch(err){
      alert('Generate failed: '+err.message)
    } finally{ setLoading(false) }
  }

  function exportTxt(){
    const lines = questions.map((q,i)=>`${i+1}. [${q.type}] ${q.question}\nAnswer: ${q.answer}\n`).join('\n')
    const blob = new Blob([lines], {type:'text/plain'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href=url; a.download='questions.txt'; a.click(); URL.revokeObjectURL(url)
  }

  function exportPdf(){
    const doc = new jsPDF()
    let y = 16
    doc.setFontSize(12)
    doc.text('Generated Questions', 14, 12)
    questions.forEach((q,i)=>{
      const lines = doc.splitTextToSize(`${i+1}. [${q.type}] ${q.question} \nAnswer: ${q.answer}`, 180)
      doc.text(lines, 14, y)
      y += lines.length*6 + 6
      if (y>270){ doc.addPage(); y=16 }
    })
    doc.save('questions.pdf')
  }

  return (
    <div className="app">
      <div className="card header">
        <div>
          <div className="h1">Quiz Generator — Netlify POC</div>
          <div className="small">Upload PDF / CSV / TXT or paste text, choose options, generate editable questions, take quiz, export.</div>
        </div>
      </div>

      <div className="card" style={{marginTop:14}}>
        <div className="controls">
          <input className="input" placeholder="Paste text here (or upload file)" value={textInput} onChange={e=>setTextInput(e.target.value)} />
          <input ref={fileRef} type="file" onChange={handleFile} />
          <select value={type} onChange={e=>setType(e.target.value)} className="input">
            <option value="mixed">Mixed types (MCQ/TF/Short)</option>
            <option value="mcq">Multiple choice</option>
            <option value="tf">True / False</option>
            <option value="short">Short answer</option>
          </select>
          <input className="input" placeholder="Topic (optional)" value={topic} onChange={e=>setTopic(e.target.value)} />
          <input className="input" type="number" min={1} max={50} value={count} onChange={e=>setCount(e.target.value)} />
          <button className="button" onClick={()=>generateFromServer() } disabled={loading}>{loading? 'Generating...':'Generate'}</button>
        </div>

        <div style={{marginTop:12}}>
          <strong>Tip:</strong> For best results set the topic field when your PDF/chapter covers many subjects.
        </div>

        <div className="quiz-area">
          <motion.div initial={{opacity:0}} animate={{opacity:1}} transition={{duration:0.45}}>
            {questions.length===0 ? <div className="small result">No questions yet. Generate some!</div> : (
              <div>
                <div style={{display:'flex',gap:8,marginTop:8}}>
                  <button className="button" onClick={exportTxt}>Export TXT</button>
                  <button className="button" onClick={exportPdf}>Export PDF</button>
                </div>
                {questions.map((q, idx)=> (
                  <div key={idx} className="question">
                    <div style={{display:'flex',justifyContent:'space-between'}}>
                      <div><strong>{idx+1}.</strong> <em className="small">[{q.difficulty||'mid'}-{q.type}]</em></div>
                      <div className="small">Editable</div>
                    </div>
                    <div style={{marginTop:8}}>
                      <input className="editable" value={q.question} onChange={e=>{
                        const copy=[...questions]; copy[idx].question=e.target.value; setQuestions(copy);
                      }} />
                    </div>
                    {q.options && q.options.length>0 && (
                      <div style={{marginTop:8}}>
                        {q.options.map((opt,i)=> (
                          <div key={i}><input className="editable" value={opt} onChange={e=>{
                            const copy=[...questions]; copy[idx].options[i]=e.target.value; setQuestions(copy);
                          }} /></div>
                        ))}
                      </div>
                    )}
                    <div style={{marginTop:6}} className="small">Answer: <input className="editable" value={q.answer} onChange={e=>{const copy=[...questions]; copy[idx].answer=e.target.value; setQuestions(copy)}} /></div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>

      <div className="card" style={{marginTop:12}}>
        <div className="h2">Help / About</div>
        <div className="small" style={{marginTop:8}}>
          This prototype uses Netlify Functions to call LLMs (Gemini or Hugging Face). It also includes a local fallback generator so students always get practice questions even when the external LLM is unavailable.
        </div>
      </div>

    </div>
  )
}
