const pdfParse = require('pdf-parse')
const fetch = require('node-fetch')

// helper: small deterministic fallback question generator
function fallbackGenerate(text, options){
  const sentences = text.replace(/\n+/g,' ').split(/[.?!]\s+/).filter(s=>s.length>30)
  const out = []
  const n = Math.min(options.count || 6, Math.max(3, Math.floor(sentences.length/3)))
  for(let i=0;i<n;i++){
    const s = sentences[i % sentences.length].trim()
    if (options.type === 'tf'){
      out.push({ type:'tf', difficulty: 'easy', question: s + ' (True/False?)', options:['True','False'], answer:'True' })
    } else if (options.type === 'short'){
      out.push({ type:'short', difficulty:'medium', question:`Explain: ${s}`, answer: s.slice(0, Math.min(120,s.length)) })
    } else {
      const correct = s
      const opts = [correct, correct + ' (not exactly)', 'Close but wrong', 'Opposite idea']
      out.push({ type:'mcq', difficulty: i%3===0?'hard':(i%2===0?'medium':'easy'), question:s, options:opts, answer: opts[0] })
    }
  }
  return out
}

function clampText(t, maxLen=14000){ return t.length>maxLen? t.slice(0,maxLen) : t }

exports.handler = async function(event, context){
  try{
    const body = JSON.parse(event.body || '{}')
    const source = body.source || 'text'
    const content = body.content || ''
    const opts = body.options || {}

    let text = ''
    if (source === 'pdf'){
      const buffer = Buffer.from(content, 'base64')
      const data = await pdfParse(buffer)
      text = data.text || ''
    } else {
      text = content
    }

    text = clampText(text)

    const prompt = `You are an exam question generator. Given the text below, produce a JSON object: { "questions": [ {"type":"mcq|tf|short", "difficulty":"easy|medium|hard","question":"...","options":[...],"answer":"..."} ] }\n\nTEXT:\n${text}\n\nRequirements:\n- Generate exactly ${opts.count||6} questions.\n- Types: ${opts.type||'mixed (mcq,tf,short)'}.\n- If user supplied topic: ${opts.topic||'none'}\n\nReturn only valid JSON (no extra commentary).\n`

    if (process.env.GEMINI_API_KEY && process.env.GEMINI_MODEL){
      try{
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${process.env.GEMINI_MODEL}:generateText`
        const resp = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type':'application/json', 'Authorization': `Bearer ${process.env.GEMINI_API_KEY}` },
          body: JSON.stringify({ prompt: { text: prompt } })
        })
        if (resp.ok){
          const j = await resp.json()
          const raw = j?.candidates?.[0]?.content || j?.outputs?.[0]?.content || j?.response || JSON.stringify(j)
          const m = raw.match(/\{\s*\"questions\"[\s\S]*\}/)
          const jsonText = m ? m[0] : raw
          const parsed = JSON.parse(jsonText)
          return { statusCode: 200, body: JSON.stringify(parsed) }
        }
      }catch(e){
        console.warn('Gemini failed', e.message)
      }
    }

    if (process.env.HF_API_KEY && process.env.HF_MODEL){
      try{
        const hfUrl = `https://api-inference.huggingface.co/models/${process.env.HF_MODEL}`
        const resp = await fetch(hfUrl, {
          method: 'POST', headers: { 'Content-Type':'application/json', 'Authorization': `Bearer ${process.env.HF_API_KEY}` },
          body: JSON.stringify({ inputs: prompt, parameters: { max_new_tokens: 512 } })
        })
        if (resp.ok){
          const j = await resp.json()
          const raw = Array.isArray(j) ? (j[0]?.generated_text || JSON.stringify(j)) : (j.generated_text || JSON.stringify(j))
          const m = raw.match(/\{\s*\"questions\"[\s\S]*\}/)
          const jsonText = m ? m[0] : raw
          const parsed = JSON.parse(jsonText)
          return { statusCode: 200, body: JSON.stringify(parsed) }
        }
      }catch(e){
        console.warn('Hugging Face failed', e.message)
      }
    }

    const fallback = fallbackGenerate(text, opts)
    return { statusCode: 200, body: JSON.stringify({ questions: fallback }) }

  }catch(err){
    console.error(err)
    return { statusCode: 500, body: JSON.stringify({ error: err.message }) }
  }
}
