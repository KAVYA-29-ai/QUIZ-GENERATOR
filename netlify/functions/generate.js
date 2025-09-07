// netlify/functions/generate.js
// Lightweight serverless function. No native deps required.
// Uses environment variables:
// GEMINI_API_KEY, GEMINI_MODEL
// HF_API_KEY, HF_MODEL

exports.handler = async function(event, context) {
  try {
    const body = JSON.parse(event.body || '{}');
    const action = body.action || 'generate';
    const text = (body.text || '').slice(0, 14000); // clamp to safe size
    const count = Number(body.count || 6);
    const qtype = body.type || 'mixed';
    const topic = body.topic || '';

    if (!text) return { statusCode: 400, body: JSON.stringify({ error: 'No text provided' }) };

    const promptForGeneration = (text, count, qtype, topic) => {
      return `You are a helpful exam question generator. Given the TEXT below, produce valid JSON with exactly ${count} questions in the format: {"questions":[{"type":"mcq|tf|short","difficulty":"easy|medium|hard","question":"...","options":["..."],"answer":"..."}]}. TEXT: ${text}
Requirements:
- Generate ${count} questions.
- Types: ${qtype}.
- If topic provided: focus on that topic: ${topic}.
Return only JSON.`;
    };

    const promptForSummarize = (text, topic) => {
      return `Summarize the following TEXT in 3 concise bullet points, and provide a 1-2 sentence explanation for each bullet. TEXT: ${text} Topic focus: ${topic}. Return JSON: {"summary":[{"point":"...","explain":"..."}]}`;
    };

    const prompt = action === 'summarize' ? promptForSummarize(text, topic) : promptForGeneration(text, count, qtype, topic);

    // Try Gemini (Generative Language API) if configured
    if (process.env.GEMINI_API_KEY && process.env.GEMINI_MODEL) {
      try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${process.env.GEMINI_MODEL}:generateText`;
        const res = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.GEMINI_API_KEY}`
          },
          body: JSON.stringify({ prompt: { text: prompt } })
        });
        if (res.ok) {
          const j = await res.json();
          // Locate text
          const raw = j?.candidates?.[0]?.content || j?.output || JSON.stringify(j);
          const jsonMatch = raw.match(/\{[\s\S]*\}/);
          const jsonText = jsonMatch ? jsonMatch[0] : raw;
          const parsed = JSON.parse(jsonText);
          return { statusCode: 200, body: JSON.stringify(parsed) };
        }
      } catch (e) {
        console.warn('Gemini call failed:', e.message);
      }
    }

    // Try Hugging Face inference as fallback
    if (process.env.HF_API_KEY && process.env.HF_MODEL) {
      try {
        const hfUrl = `https://api-inference.huggingface.co/models/${process.env.HF_MODEL}`;
        const res = await fetch(hfUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.HF_API_KEY}`
          },
          body: JSON.stringify({ inputs: prompt, parameters: { max_new_tokens: 512 } })
        });
        if (res.ok) {
          const j = await res.json();
          const raw = Array.isArray(j) ? (j[0]?.generated_text || JSON.stringify(j)) : (j.generated_text || JSON.stringify(j));
          const jsonMatch = raw.match(/\{[\s\S]*\}/);
          const jsonText = jsonMatch ? jsonMatch[0] : raw;
          const parsed = JSON.parse(jsonText);
          return { statusCode: 200, body: JSON.stringify(parsed) };
        }
      } catch (e) {
        console.warn('HuggingFace call failed:', e.message);
      }
    }

    // Final local deterministic fallback if both calls fail
    function fallbackGenerate(text, count, qtype) {
      const sentences = text.replace(/\n+/g,' ').split(/[.?!]\s+/).filter(s=>s.length>30);
      const n = Math.max(3, Math.min(count, Math.floor(sentences.length/2) || 3));
      const out = [];
      for (let i=0;i<n;i++) {
        const s = sentences[i % sentences.length].trim();
        if (qtype === 'tf') out.push({ type:'tf', difficulty:'easy', question: s + ' (True/False?)', options:['True','False'], answer:'True' });
        else if (qtype === 'short') out.push({ type:'short', difficulty:'medium', question: 'Explain: ' + s.slice(0,140), answer: s.slice(0,120) });
        else {
          out.push({ type:'mcq', difficulty: (i%3===0?'hard':(i%2===0?'medium':'easy')), question: s, options: [s, s + ' (close)', 'Opposite idea', 'Not this'], answer: s });
        }
      }
      return { questions: out };
    }

    const fallback = fallbackGenerate(text, count, qtype);
    return { statusCode: 200, body: JSON.stringify(fallback) };

  } catch (err) {
    console.error('Function error', err);
    return { statusCode: 500, body: JSON.stringify({ error: err.message || 'Server error' }) };
  }
};
