const { GoogleGenerativeAI } = require('@google/generative-ai');
const pdf = require('pdf-parse');

exports.handler = async (event, context) => {
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Content-Type': 'application/json'
  };

  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers, body: '' };
  }

  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, headers, body: JSON.stringify({ error: 'Method not allowed' }) };
  }

  try {
    const { pdfFile, type, difficulty, questionCount, topic } = JSON.parse(event.body);

    let content = '';
    let images = [];

    if (pdfFile) {
      const dataBuffer = Buffer.from(pdfFile, 'base64');
      const data = await pdf(dataBuffer, { 
        pagerender: page => page.getTextContent().then(tc => tc.items.map(i => i.str).join(' ')) 
      });
      content = data.text;

      // images extraction optional: complex in pure Node, can use external libs later
      // for now Gemini will only use text
    }

    // Primary AI
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    const prompts = {
      quiz: `Generate ${questionCount} quiz questions from the following content with ${difficulty} difficulty level. Content: ${content}. Include image references if any. Return JSON with questions array.`,
      summary: `Summarize and explain the following content: ${content}`,
      extract: `Extract information about "${topic}" from the following content: ${content}`
    };

    let prompt = prompts[type] || prompts.quiz;

    let result;
    try {
      const response = await model.generateContent(prompt);
      result = response.response.text();
    } catch (geminiError) {
      console.log('Gemini failed, using fallback...');
      if (type === 'quiz') result = generateFallbackQuiz(content, questionCount, difficulty);
      else if (type === 'summary') result = generateFallbackSummary(content);
      else if (type === 'extract') result = generateFallbackExtraction(content, topic);
    }

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ success: true, data: result, timestamp: new Date().toISOString() })
    };
  } catch (error) {
    console.error('Function error:', error);
    return { statusCode: 500, headers, body: JSON.stringify({ error: 'Failed', message: error.message }) };
  }
};

// Fallback functions
function generateFallbackQuiz(content, questionCount, difficulty) {
  const words = content.split(' ').filter(w => w.length > 3);
  const questions = [];
  for (let i = 0; i < Math.min(questionCount, 10); i++) {
    const randomWord = words[Math.floor(Math.random() * words.length)];
    const q = {
      id: i + 1,
      type: i % 3 === 0 ? 'mcq' : (i % 3 === 1 ? 'truefalse' : 'short'),
      question: `What is the significance of "${randomWord}"?`,
      difficulty
    };
    if (q.type === 'mcq') {
      q.options = ['A) Key concept', 'B) Irrelevant', 'C) Supporting detail', 'D) Example'];
      q.correct = 'A';
    } else if (q.type === 'truefalse') {
      q.options = ['True', 'False']; q.correct = 'True';
    }
    q.explanation = 'Related to main content.';
    questions.push(q);
  }
  return JSON.stringify({ questions });
}

function generateFallbackSummary(content) {
  const sentences = content.split('.').filter(s => s.trim().length > 10);
  const summary = sentences.slice(0, 3).join('. ') + '.';
  return `## Summary\n\n${summary}\n\n## Key Points\n• Main concepts\n• Important details\n• Relevant info\n\n## Explanation\nContent covers various interconnected topics.`;
}

function generateFallbackExtraction(content, topic) {
  const relevant = content.split('.').filter(s => s.toLowerCase().includes(topic.toLowerCase()));
  return `## Info on "${topic}"\n\n${relevant.join('. ')}\n\n## Analysis\nRelevant details from content.`;
}
