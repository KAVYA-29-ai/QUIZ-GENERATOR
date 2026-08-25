const { GoogleGenerativeAI } = require('@google/generative-ai');

const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Content-Type': 'application/json',
};

exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS') return { statusCode: 204, headers, body: '' };
  if (event.httpMethod !== 'POST') return json(405, { success: false, error: 'Method not allowed' });

  try {
    const body = safeJson(event.body);
    const normalized = normalizeRequest(body);

    if (!normalized.content) return json(400, { success: false, error: 'Add some text or upload a file first.' });
    if (!normalized.apiKey) return json(500, { success: false, error: 'GEMINI_API_KEY is not configured in Netlify.' });

    const result = await generateWithGemini(normalized);
    return json(200, { success: true, data: result, ...result });
  } catch (error) {
    console.error('Quiz generator error:', error);
    return json(500, { success: false, error: friendlyError(error) });
  }
};

function normalizeRequest(body) {
  const legacy = body && body.action === 'generate';
  const content = String(body?.content ?? body?.text ?? '').trim();
  const count = clamp(Number(body?.config?.count ?? body?.count ?? 10), 3, 30);
  const requestedType = body?.config?.type ?? body?.type ?? 'mixed';

  const questionTypes = body?.config?.questionTypes || buildQuestionTypes(count, requestedType);
  const difficultyDistribution = body?.config?.difficultyDistribution || { easy: 35, medium: 40, hard: 20, expert: 5 };

  return {
    content: content.slice(0, 30000),
    topic: String(body?.config?.topic ?? body?.topic ?? '').trim(),
    questionTypes,
    difficultyDistribution,
    apiKey: process.env.GEMINI_API_KEY,
    legacy,
  };
}

async function generateWithGemini({ content, topic, questionTypes, difficultyDistribution, apiKey }) {
  const total = Object.values(questionTypes).reduce((sum, value) => sum + Number(value || 0), 0) || 10;
  const modelName = process.env.GEMINI_MODEL || 'gemini-3.7-flash';
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: modelName });

  const prompt = `You are QuizGen Pro, an expert educational assessment designer.
Create high-quality study material from the source content below.

${topic ? `TOPIC: ${topic}\n` : ''}SOURCE:
${content}

Return JSON only. No markdown fences and no commentary.

Schema:
{
  "questions": [
    {
      "id": number,
      "type": "mcq" | "truefalse" | "short",
      "question": string,
      "options": string[],
      "correct": "A" | "B" | "C" | "D" | "True" | "False",
      "difficulty": "easy" | "medium" | "hard" | "expert",
      "explanation": string,
      "sampleAnswer": string
    }
  ],
  "summary": string,
  "flashcards": [{ "id": number, "front": string, "back": string, "category": string }]
}

Rules:
- Generate exactly ${total} questions.
- Type distribution: ${JSON.stringify(questionTypes)}.
- Difficulty distribution: ${JSON.stringify(difficultyDistribution)}.
- MCQ questions must have exactly 4 options and correct must be A/B/C/D.
- True/False questions must have exactly ["True", "False"].
- Short questions need a useful sampleAnswer.
- Explanations must be concise but educational.
- Generate 12 useful flashcards.
- Base every question on the source. Do not invent facts unrelated to it.`;

  const response = await model.generateContent(prompt);
  const text = response.response.text();
  const parsed = parseJson(text);
  return sanitizeResult(parsed, content, questionTypes, total);
}

function sanitizeResult(parsed, content, questionTypes, total) {
  const rawQuestions = Array.isArray(parsed?.questions) ? parsed.questions : [];
  const questions = rawQuestions.slice(0, total).map((q, index) => {
    const type = normalizeType(q?.type);
    const options = type === 'mcq'
      ? normalizeOptions(q?.options, 4)
      : type === 'truefalse' ? ['True', 'False'] : [];

    let correct = String(q?.correct ?? '').trim();
    if (type === 'mcq' && !['A', 'B', 'C', 'D'].includes(correct)) correct = 'A';
    if (type === 'truefalse' && !['True', 'False'].includes(correct)) correct = 'True';

    return {
      id: index + 1,
      type,
      question: String(q?.question || `Question ${index + 1}`),
      options,
      correct,
      answer: type === 'mcq' ? options['ABCD'.indexOf(correct)] : correct,
      difficulty: ['easy', 'medium', 'hard', 'expert'].includes(q?.difficulty) ? q.difficulty : 'medium',
      explanation: String(q?.explanation || 'Review the source material for the reasoning behind this answer.'),
      sampleAnswer: type === 'short' ? String(q?.sampleAnswer || '') : '',
    };
  });

  const completed = questions.length >= total ? questions : fillFallbackQuestions(questions, content, questionTypes, total);
  const flashcards = Array.isArray(parsed?.flashcards) ? parsed.flashcards.slice(0, 12).map((card, index) => ({
    id: index + 1,
    front: String(card?.front || 'Key concept'),
    back: String(card?.back || 'Review this concept in the source material.'),
    category: String(card?.category || 'General'),
  })) : [];

  return {
    questions: completed,
    summary: String(parsed?.summary || 'Study the generated questions and flashcards to reinforce the source material.'),
    flashcards: flashcards.length ? flashcards : buildFallbackFlashcards(content),
  };
}

function fillFallbackQuestions(existing, content, questionTypes, total) {
  const sentences = content.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 25);
  const words = content.split(/\s+/).map(w => w.replace(/[^A-Za-z0-9-]/g, '')).filter(w => w.length > 4);
  const result = [...existing];
  let i = 0;
  while (result.length < total) {
    const source = sentences[i % Math.max(sentences.length, 1)] || content.slice(0, 180);
    const word = words[i % Math.max(words.length, 1)] || 'concept';
    const types = Object.keys(questionTypes).filter(t => Number(questionTypes[t]) > 0);
    const kind = types[i % Math.max(types.length, 1)] || 'mcq';
    const type = normalizeType(kind);
    result.push({
      id: result.length + 1,
      type,
      question: type === 'short' ? `Explain the idea described here: ${source}` : `According to the source, what is the best interpretation of "${word}"?`,
      options: type === 'mcq' ? [source, `${word} is unrelated`, 'None of the above', 'The source does not discuss it'] : type === 'truefalse' ? ['True', 'False'] : [],
      correct: type === 'truefalse' ? 'True' : 'A',
      answer: type === 'short' ? source : type === 'truefalse' ? 'True' : source,
      difficulty: 'medium',
      explanation: 'Fallback question generated from the supplied source text.',
      sampleAnswer: type === 'short' ? source : '',
    });
    i++;
  }
  return result.slice(0, total);
}

function buildFallbackFlashcards(content) {
  const chunks = content.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean).slice(0, 12);
  return chunks.map((chunk, index) => ({
    id: index + 1,
    front: `Concept ${index + 1}`,
    back: chunk.slice(0, 300),
    category: 'Source',
  }));
}

function buildQuestionTypes(count, type) {
  if (type === 'mcq') return { mcq: count };
  if (type === 'truefalse' || type === 'tf') return { truefalse: count };
  if (type === 'short') return { short: count };
  const mcq = Math.ceil(count * 0.5);
  const tf = Math.floor(count * 0.25);
  return { mcq, truefalse: tf, short: count - mcq - tf };
}

function normalizeType(type) {
  if (type === 'tf' || type === 'true_false' || type === 'truefalse') return 'truefalse';
  if (type === 'short' || type === 'shortanswer') return 'short';
  return 'mcq';
}

function normalizeOptions(options, length) {
  const fallback = ['Option A', 'Option B', 'Option C', 'Option D'];
  return Array.isArray(options) && options.length === length ? options.map(String) : fallback;
}

function parseJson(text) {
  const cleaned = String(text || '').replace(/^```(?:json)?/i, '').replace(/```$/i, '').trim();
  try { return JSON.parse(cleaned); } catch (_) {
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    if (start >= 0 && end > start) return JSON.parse(cleaned.slice(start, end + 1));
    throw new Error('Gemini returned an invalid JSON response.');
  }
}

function safeJson(value) {
  try { return typeof value === 'string' ? JSON.parse(value) : (value || {}); }
  catch (_) { throw new Error('Invalid request body.'); }
}

function clamp(value, min, max) { return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : min; }
function json(statusCode, body) { return { statusCode, headers, body: JSON.stringify(body) }; }
function friendlyError(error) {
  const message = String(error?.message || error);
  if (/API key|api_key|GEMINI_API_KEY/i.test(message)) return 'Gemini API configuration is missing or invalid.';
  if (/404|not found|model/i.test(message)) return 'The Gemini model is unavailable. Set GEMINI_MODEL in Netlify environment variables if needed.';
  return message.slice(0, 500);
}
