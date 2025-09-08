// netlify/functions/generate.js
const { GoogleGenerativeAI } = require('@google/generative-ai');

exports.handler = async (event, context) => {
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
  };

  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers, body: '' };
  }

  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Method not allowed' }),
    };
  }

  try {
    const { content, type, config } = JSON.parse(event.body);

    if (!content || !type) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({
          success: false,
          error: 'Missing required parameters: content and type',
        }),
      };
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error('GEMINI_API_KEY not set');

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

    let result = {};

    if (type === 'questions') {
      const { difficultyDistribution = {}, questionTypes = {} } = config || {};
      const expectedCount = Object.values(questionTypes).reduce((a, b) => a + b, 0) || 10;

      const prompt = `
        Based on the following content, generate comprehensive learning materials:

        CONTENT:
        ${content.substring(0, 8000)}

        REQUIREMENTS:
        1. Generate exactly ${expectedCount} questions total (do not exceed this count).
        2. Distribute by difficulty: ${JSON.stringify(difficultyDistribution)}
        3. Distribute by type: ${JSON.stringify(questionTypes)}
        4. Provide a comprehensive summary
        5. Generate 12 flashcards

        RESPONSE FORMAT (JSON only):
        {
          "questions": [
            {
              "id": number,
              "type": "mcq|truefalse|short",
              "question": "string",
              "options": ["Option A", "Option B", "Option C", "Option D"],
              "correct": "A|B|C|D or True/False depending on type",
              "difficulty": "easy|medium|hard|expert",
              "explanation": "string",
              "sampleAnswer": "only for short answers"
            }
          ],
          "summary": "string",
          "flashcards": [
            {
              "id": number,
              "front": "term",
              "back": "explanation",
              "category": "string",
              "color": "hex code"
            }
          ]
        }

        IMPORTANT:
        - For MCQs: always provide 4 options (A-D) and correct must be A, B, C, or D only.
        - For True/False: options must be ["True", "False"] and correct must be "True" or "False".
        - Do not exceed ${expectedCount} questions total.
        - Ensure output is valid JSON only (no markdown).
      `;

      const geminiResult = await model.generateContent(prompt);
      const response = await geminiResult.response;
      const text = response.text();

      try {
        const cleanedText = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        const parsed = JSON.parse(cleanedText);

        result = {
          questions: parsed.questions || [],
          summary: parsed.summary || 'No summary generated.',
          flashcards: parsed.flashcards || [],
        };

        if (result.questions.length > expectedCount) {
          result.questions = result.questions.slice(0, expectedCount);
        }

        result.questions = result.questions.map((q, i) => {
          const base = {
            id: q.id || i + 1,
            type: q.type || 'mcq',
            question: q.question || 'Sample question',
            options: q.options || ['Option A', 'Option B', 'Option C', 'Option D'],
            correct: q.correct || 'A',
            difficulty: q.difficulty || 'medium',
            explanation: q.explanation || 'No explanation provided.',
          };
          if (base.type === 'mcq') {
            if (!Array.isArray(base.options) || base.options.length !== 4) {
              base.options = ['Option A', 'Option B', 'Option C', 'Option D'];
            }
            if (!['A', 'B', 'C', 'D'].includes(base.correct)) {
              base.correct = 'A';
            }
          } else if (base.type === 'truefalse') {
            base.options = ['True', 'False'];
            if (!['True', 'False'].includes(base.correct)) {
              base.correct = 'True';
            }
          } else if (base.type === 'short') {
            base.sampleAnswer = q.sampleAnswer || 'Sample answer';
          }
          return base;
        });

        result.flashcards = result.flashcards.map((c, i) => ({
          id: c.id || i + 1,
          front: c.front || 'Concept',
          back: c.back || 'Definition',
          category: c.category || 'General',
          color: c.color || '#4f46e5',
        }));
      } catch (e) {
        console.error('JSON parse error:', e);
        result = generateFallbackContent(content, config, expectedCount);
      }
    }

    return { statusCode: 200, headers, body: JSON.stringify({ success: true, data: result }) };
  } catch (err) {
    console.error('Function error:', err);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ success: false, error: err.message }),
    };
  }
};

function generateFallbackContent(content, config, expectedCount) {
  const { questionTypes = { mcq: 6, short: 2, truefalse: 2 } } = config || {};
  const words = content.split(/\s+/).filter((w) => w.length > 4);
  const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 20);
  const questions = [];
  let id = 1;

  Object.entries(questionTypes).forEach(([type, count]) => {
    for (let i = 0; i < count && id <= expectedCount; i++) {
      const keyWord = words[Math.floor(Math.random() * words.length)] || 'concept';
      if (type === 'mcq') {
        questions.push({
          id: id++,
          type: 'mcq',
          question: `What is the significance of "${keyWord}"?`,
          options: [
            `${keyWord} is a key concept`,
            `${keyWord} is a detail`,
            `${keyWord} is an example`,
            `${keyWord} is unrelated`,
          ],
          correct: 'A',
          difficulty: 'easy',
          explanation: `${keyWord} appears as a key concept.`,
        });
      } else if (type === 'truefalse') {
        const isTrue = Math.random() > 0.5;
        questions.push({
          id: id++,
          type: 'truefalse',
          question: `The material discusses "${keyWord}" as a main theme.`,
          options: ['True', 'False'],
          correct: isTrue ? 'True' : 'False',
          difficulty: 'medium',
          explanation: `This statement is ${isTrue ? 'true' : 'false'}.`,
        });
      } else if (type === 'short') {
        questions.push({
          id: id++,
          type: 'short',
          question: `Explain the role of "${keyWord}".`,
          difficulty: 'hard',
          explanation: `Analyze ${keyWord} with examples.`,
          sampleAnswer: `${keyWord} is important for understanding the material.`,
        });
      }
    }
  });

  return {
    questions,
    summary: `Generated ${questions.length} fallback questions.`,
    flashcards: [],
  };
}
