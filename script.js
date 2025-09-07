// script.js
// Frontend logic: extract text (PDF.js), call server function, manage quiz + streak, export.

const pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js';

const fileInput = document.getElementById('fileInput');
const textArea = document.getElementById('textArea');
const generateBtn = document.getElementById('generateBtn');
const qType = document.getElementById('qType');
const qCount = document.getElementById('qCount');
const topicInput = document.getElementById('topic');
const statusDiv = document.getElementById('status');
const questionsContainer = document.getElementById('questionsContainer');
const startQuizBtn = document.getElementById('startQuizBtn');
const exportTxtBtn = document.getElementById('exportTxtBtn');
const exportPdfBtn = document.getElementById('exportPdfBtn');
const quizControls = document.getElementById('quizControls');
const quizArea = document.getElementById('quizArea');
const reportArea = document.getElementById('reportArea');
const resultsSection = document.getElementById('results');

const helpBtn = document.getElementById('helpBtn');
const helpModal = document.getElementById('helpModal');
const closeHelp = document.getElementById('closeHelp');

let questions = []; // list of {type,difficulty,question,options,answer}
let quizState = null; // {index,score,streak,order}

function animateIntro() {
  if (window.anime) {
    anime({
      targets: '.card',
      translateY: [-8,0],
      opacity: [0,1],
      duration: 600,
      easing: 'spring(1,80,10,0)'
    });
  }
}
animateIntro();

// Help modal events
helpBtn.addEventListener('click', ()=> helpModal.setAttribute('aria-hidden','false'));
closeHelp.addEventListener('click', ()=> helpModal.setAttribute('aria-hidden','true'));

// File handling
fileInput.addEventListener('change', async (e)=> {
  const f = e.target.files[0];
  if (!f) return;
  const ext = f.name.split('.').pop().toLowerCase();
  if (ext === 'pdf') {
    statusDiv.textContent = 'Extracting PDF text locally...';
    const txt = await extractTextFromPDF(f);
    textArea.value = txt.slice(0, 20000); // cap in UI
    statusDiv.textContent = 'PDF text extracted. You can edit or press Generate.';
  } else {
    const txt = await f.text();
    textArea.value = txt;
    statusDiv.textContent = 'File loaded. Edit text or press Generate.';
  }
});

// PDF.js extraction
async function extractTextFromPDF(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({data: arrayBuffer}).promise;
  let full = '';
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const txtContent = await page.getTextContent();
    const strs = txtContent.items.map(i => i.str);
    full += strs.join(' ') + '\\n\\n';
    // keep under a limit
    if (full.length > 200000) break;
  }
  return full;
}

// Generate button
generateBtn.addEventListener('click', async ()=> {
  const rawText = textArea.value.trim();
  if (!rawText) {
    statusDiv.textContent = 'Please paste text or upload a file first.';
    return;
  }
  const payload = {
    action: 'generate',
    text: rawText,
    count: Number(qCount.value || 8),
    type: qType.value || 'mixed',
    topic: (topicInput.value || '').trim()
  };
  statusDiv.textContent = 'Generating questions (Gemini -> HF -> fallback)...';
  generateBtn.disabled = true;
  try {
    const res = await fetch('/.netlify/functions/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await res.json();
    if (!res.ok) throw new Error(j?.error || 'Server error');
    questions = j.questions || [];
    if (!Array.isArray(questions)) questions = [];
    persistQuestions();
    renderQuestionsList();
    statusDiv.textContent = `Generated ${questions.length} questions. Edit as needed, then Start Quiz.`;
    quizControls.style.display = questions.length ? 'flex' : 'none';
  } catch (err) {
    statusDiv.textContent = 'Generate failed: ' + err.message;
    // fallback: try to build basic local fallback (in-client)
    questions = localFallback(rawText, payload);
    persistQuestions();
    renderQuestionsList();
    statusDiv.textContent = `Used local fallback generator — ${questions.length} questions ready.`;
    quizControls.style.display = questions.length ? 'flex' : 'none';
  } finally {
    generateBtn.disabled = false;
  }
});

// render editable list
function renderQuestionsList(){
  questionsContainer.innerHTML = '';
  if (!questions || questions.length===0) {
    questionsContainer.innerHTML = '<div class="muted">No questions generated yet.</div>';
    return;
  }
  questions.forEach((q, idx) => {
    const div = document.createElement('div');
    div.className = 'question-card';
    div.innerHTML = `
      <div><strong>${idx+1}.</strong> <span class="muted">[${q.difficulty||'mid'} • ${q.type}]</span></div>
      <div style="margin-top:8px"><input data-idx="${idx}" data-field="question" class="edit-input" value="${escapeHtml(q.question)}" /></div>
      <div class="options" id="opts-${idx}"></div>
      <div class="answer small">Answer: <input data-idx="${idx}" data-field="answer" class="edit-input" value="${escapeHtml(q.answer||'')}" /></div>
      <div style="margin-top:8px" class="row" id="tagrow-${idx}">
        <button data-action="mark-good" data-idx="${idx}" class="btn small">Mark Good</button>
        <button data-action="mark-bad" data-idx="${idx}" class="btn small">Mark Bad</button>
      </div>
    `;
    questionsContainer.appendChild(div);

    const optsContainer = document.getElementById('opts-'+idx);
    if (q.options && q.options.length) {
      q.options.forEach((opt, oi) => {
        const oiEl = document.createElement('div');
        oiEl.innerHTML = `<input data-idx="${idx}" data-field="opt-${oi}" class="edit-input" value="${escapeHtml(opt)}" />`;
        optsContainer.appendChild(oiEl);
      });
    }

    // wire edit inputs
    Array.from(div.querySelectorAll('.edit-input')).forEach(inp => {
      inp.addEventListener('input', (ev) => {
        const i = Number(ev.target.dataset.idx);
        const field = ev.target.dataset.field;
        if (field.startsWith('opt-')) {
          const oi = Number(field.split('-')[1]);
          questions[i].options[oi] = ev.target.value;
        } else {
          questions[i][field] = ev.target.value;
        }
        persistQuestions();
      });
    });

    // mark good/bad
    div.querySelectorAll('button[data-action]').forEach(b => {
      b.addEventListener('click', (ev) => {
        const idx = Number(ev.target.dataset.idx);
        const action = ev.target.dataset.action;
        questions[idx].meta = questions[idx].meta || {};
        questions[idx].meta.quality = action === 'mark-good' ? 'good' : 'bad';
        persistQuestions();
        // small animation
        if (window.anime) anime({
          targets: div, translateX: [0, -6, 0], duration: 380
        });
      });
    });
  });
}

// Start quiz flow
startQuizBtn.addEventListener('click', () => {
  if (!questions || questions.length===0) return;
  startQuiz();
});

function startQuiz(){
  quizState = {
    index: 0,
    score: 0,
    streak: 0,
    order: questions.map((_,i)=>i)
  };
  // shuffle
  quizState.order = shuffleArray(quizState.order);
  showQuizUI();
}

function showQuizUI(){
  quizArea.style.display = 'block';
  reportArea.style.display = 'none';
  quizArea.innerHTML = '';
  renderQuizQuestion();
}

function renderQuizQuestion(){
  const qi = quizState.index;
  if (qi >= quizState.order.length) {
    return finishQuiz();
  }
  const q = questions[ quizState.order[qi] ];
  // adapt difficulty: if streak >=2 increase difficulty label (affects selection only)
  const effectiveDifficulty = (quizState.streak >=2 && q.difficulty !== 'hard') ? 'hard' : q.difficulty || 'medium';
  quizArea.innerHTML = `
    <div class="card quiz-wrap">
      <div class="quiz-question">Q${qi+1}. ${escapeHtml(q.question)}</div>
      <div class="streak">Score: ${quizState.score} • Streak: ${quizState.streak}</div>
      <div class="options" id="quizOpts"></div>
      <div style="margin-top:10px" id="quizMsg"></div>
    </div>
  `;
  const optsDiv = document.getElementById('quizOpts');
  if (q.type === 'mcq' && q.options && q.options.length) {
    q.options.forEach((opt, i) => {
      const but = document.createElement('button');
      but.className = 'btn';
      but.style.marginBottom = '6px';
      but.textContent = opt;
      but.addEventListener('click', ()=> handleAnswer(opt));
      optsDiv.appendChild(but);
    });
  } else if (q.type === 'tf') {
    ['True','False'].forEach(v=>{
      const but = document.createElement('button'); but.className='btn'; but.textContent=v;
      but.addEventListener('click', ()=> handleAnswer(v));
      optsDiv.appendChild(but);
    });
  } else { // short answer
    const inp = document.createElement('input'); inp.type='text'; inp.placeholder='Type your short answer'; inp.style.width='80%';
    const but = document.createElement('button'); but.className='btn'; but.textContent='Submit';
    but.addEventListener('click', ()=> handleAnswer(inp.value));
    optsDiv.appendChild(inp); optsDiv.appendChild(but);
  }
}

function handleAnswer(given){
  const curIndex = quizState.order[quizState.index];
  const q = questions[curIndex];
  const correct = normalizeText(q.answer || '');
  const g = normalizeText(String(given||''));
  const isCorrect = (q.type === 'short') ? (g.length && (correct.includes(g) || g.includes(correct) || similarity(g,correct) > 0.66)) : (g === normalizeText(q.answer));
  const msg = document.getElementById('quizMsg') || document.createElement('div');
  if (isCorrect){
    quizState.score += 1;
    quizState.streak += 1;
    msg.textContent = 'Correct ✓';
    msg.style.color = '#8efc8e';
  } else {
    quizState.streak = 0;
    msg.textContent = 'Incorrect ✕  (Answer: ' + (q.answer||'') + ')';
    msg.style.color = '#ff9b9b';
  }
  quizState.index += 1;

  // small animation
  if (window.anime) anime({
    targets: '#quizMsg', translateY: [-6,0], opacity: [0,1], duration: 300
  });

  if (quizState.index < quizState.order.length) {
    setTimeout(()=> renderQuizQuestion(), 700);
  } else {
    setTimeout(()=> finishQuiz(), 700);
  }
}

function finishQuiz(){
  quizArea.style.display = 'none';
  reportArea.style.display = 'block';
  const total = quizState.order.length;
  const score = quizState.score;
  reportArea.innerHTML = `<h3>Result</h3>
    <p>Score: <strong>${score}/${total}</strong></p>
    <p>Accuracy: <strong>${Math.round((score/total)*100)}%</strong></p>
    <p>Max streak (approx): <strong>${quizState.streak}</strong></p>
    <div style="margin-top:8px" class="row">
      <button id="reviewBtn" class="btn">Review Questions</button>
    </div>
  `;
  document.getElementById('reviewBtn').addEventListener('click', ()=> {
    // show editable list again
    questionsContainer.scrollIntoView({behavior:'smooth'});
  });
  // Persist a simple performance history in localStorage
  const history = JSON.parse(localStorage.getItem('qq_history')||'[]');
  history.unshift({date: new Date().toISOString(), total, score});
  localStorage.setItem('qq_history', JSON.stringify(history.slice(0,50)));
}

// Export TXT
exportTxtBtn.addEventListener('click', ()=>{
  const lines = questions.map((q,i)=>`${i+1}. [${q.type}] ${q.question}\\nAnswer: ${q.answer||''}\\n`).join('\\n');
  const blob = new Blob([lines], {type:'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'questions.txt'; a.click(); URL.revokeObjectURL(url);
});

// Export PDF using jsPDF
exportPdfBtn.addEventListener('click', ()=>{
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({unit:'pt', format:'a4'});
  doc.setFontSize(12);
  let y = 40;
  doc.text('Generated Questions', 40, 28);
  questions.forEach((q,i)=>{
    const text = `${i+1}. [${q.type}] ${q.question}\\nAnswer: ${q.answer||''}`;
    const lines = doc.splitTextToSize(text, 500);
    doc.text(lines, 40, y);
    y += lines.length * 14 + 8;
    if (y > 740) { doc.addPage(); y = 40; }
  });
  doc.save('questions.pdf');
});

// local fallback generator (client-side) if server fails
function localFallback(text, opts){
  const sentences = text.replace(/\\n+/g,' ').split(/[.?!]\\s+/).filter(s=>s.length>30);
  const out = [];
  const count = Math.max(3, Math.min(opts.count||6, Math.floor(sentences.length/2)));
  for (let i=0;i<count;i++){
    const s = sentences[i % sentences.length].trim();
    if (opts.type === 'tf') {
      out.push({type:'tf', difficulty:'easy', question: s + ' (True/False?)', options:['True','False'], answer:'True'});
    } else if (opts.type === 'short') {
      out.push({type:'short', difficulty:'medium', question:`Explain: ${s}`, answer: s.slice(0,120)});
    } else {
      out.push({type:'mcq', difficulty: i%3===0? 'hard': (i%2===0? 'medium':'easy'), question: s, options: [s, s + ' (close)', 'Opposite idea', 'Not this'], answer: s});
    }
  }
  return out;
}

// persist to localStorage
function persistQuestions(){ localStorage.setItem('qq_questions', JSON.stringify(questions)); }

// load persisted on start
(function loadPersisted(){
  try {
    const s = localStorage.getItem('qq_questions');
    if (s) { questions = JSON.parse(s); renderQuestionsList(); quizControls.style.display = questions.length ? 'flex' : 'none'; }
  } catch(e){}
})();

// small helpers
function escapeHtml(s){ return (s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function shuffleArray(arr){ const a = arr.slice(); for(let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]] } return a; }
function normalizeText(s){ return (s||'').toLowerCase().replace(/[^a-z0-9 ]/g,'').trim(); }
// similarity simple (dice's coefficient)
function similarity(a,b){
  if (!a || !b) return 0;
  a = a.split(' '); b = b.split(' ');
  const inter = a.filter(x=> b.includes(x)).length;
  return (2*inter)/(a.length + b.length);
}
