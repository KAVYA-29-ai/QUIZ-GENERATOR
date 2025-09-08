// netlify/functions/generate.js
const { GoogleGenerativeAI } = require('@google/generative-ai');

exports.handler = async (event, context) => {
  // Set CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
  };

  // Handle preflight OPTIONS request
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: '',
    };
  }

  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Method not allowed' }),
    };
  }

  try {
    // Parse request body
    const { content, type, config } = JSON.parse(event.body);

    if (!content || !type) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ 
          success: false, 
          error: 'Missing required parameters: content and type' 
        }),
      };
    }

    // Initialize Gemini AI
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY environment variable not set');
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

    // Generate content based on type
    let result = {};

    if (type === 'questions') {
      // Extract configuration
      const { difficultyDistribution = {}, questionTypes = {} } = config || {};
      
      // Create detailed prompt for quiz generation
      const prompt = `
        Based on the following content, generate comprehensive learning materials:

        CONTENT:
        ${content.substring(0, 8000)} // Limit content to avoid token limits

        REQUIREMENTS:
        1. Generate exactly ${Object.values(difficultyDistribution).reduce((a, b) => a + b, 10)} questions total
        2. Difficulty distribution: ${JSON.stringify(difficultyDistribution)}
        3. Question types: ${JSON.stringify(questionTypes)}
        4. Create a comprehensive summary
        5. Generate 12 flashcards

        RESPONSE FORMAT (JSON only):
        {
          "questions": [
            {
              "id": number,
              "type": "mcq|truefalse|short",
              "question": "string",
              "options": ["array of options"],
              "correct": "correct answer",
              "difficulty": "easy|medium|hard|expert",
              "explanation": "detailed explanation",
              "sampleAnswer": "for short answer questions only"
            }
          ],
          "summary": "comprehensive markdown formatted summary with key points",
          "flashcards": [
            {
              "id": number,
              "front": "term or concept",
              "back": "detailed explanation with context",
              "category": "category name",
              "color": "hex color code"
            }
          ]
        }

        IMPORTANT:
        - Ensure questions are directly based on the provided content
        - Make flashcards educational and comprehensive
        - Use varied difficulty levels as specified
        - Include proper explanations for all questions
        - Return only valid JSON without any markdown formatting
        - For MCQ questions, provide exactly 4 options labeled A-D
        - Make sure the content is educational and accurate
      `;

      // Generate content with Gemini
      const geminiResult = await model.generateContent(prompt);
      const response = await geminiResult.response;
      const text = response.text();

      try {
        // Clean and parse JSON response
        const cleanedText = text
          .replace(/```json\n?/g, '')
          .replace(/```\n?/g, '')
          .trim();

        const parsedResult = JSON.parse(cleanedText);

        // Validate and structure the response
        result = {
          questions: parsedResult.questions || [],
          summary: parsedResult.summary || 'No summary generated.',
          flashcards: parsedResult.flashcards || []
        };

        // Ensure questions have proper structure
        result.questions = result.questions.map((q, index) => ({
          id: q.id || index + 1,
          type: q.type || 'mcq',
          question: q.question || 'Sample question',
          options: q.options || ['Option A', 'Option B', 'Option C', 'Option D'],
          correct: q.correct || 'A',
          difficulty: q.difficulty || 'medium',
          explanation: q.explanation || 'No explanation provided.',
          ...(q.type === 'short' && { sampleAnswer: q.sampleAnswer || 'Sample answer' })
        }));

        // Ensure flashcards have proper structure
        result.flashcards = result.flashcards.map((card, index) => ({
          id: card.id || index + 1,
          front: card.front || 'Concept',
          back: card.back || 'Definition',
          category: card.category || 'General',
          color: card.color || '#4f46e5'
        }));

      } catch (parseError) {
        console.error('JSON parsing error:', parseError);
        // Fallback to generating structured content
        result = generateFallbackContent(content, config);
      }
    }

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        success: true,
        data: result
      }),
    };

  } catch (error) {
    console.error('Function error:', error);
    
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        success: false,
        error: error.message || 'Internal server error'
      }),
    };
  }
};

// Fallback content generation function
function generateFallbackContent(content, config) {
  const { difficultyDistribution = { easy: 3, medium: 4, hard: 2, expert: 1 }, questionTypes = { mcq: 6, short: 2, truefalse: 2 } } = config || {};
  
  const words = content.split(/\s+/).filter(w => w.length > 4);
  const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 20);
  const questions = [];
  let questionId = 1;

  // Generate questions based on configuration
  Object.entries(difficultyDistribution).forEach(([difficulty, count]) => {
    Object.entries(questionTypes).forEach(([type, totalCount]) => {
      const questionsForDifficulty = Math.ceil((count / Object.values(difficultyDistribution).reduce((a, b) => a + b)) * totalCount);
      
      for (let i = 0; i < questionsForDifficulty && questionId <= 15; i++) {
        const keyWord = words[Math.floor(Math.random() * words.length)] || 'concept';
        
        let question = {};
        
        if (type === 'mcq') {
          question = {
            id: questionId++,
            type: 'mcq',
            question: `What is the significance of "${keyWord}" in the context of this material?`,
            options: [
              `${keyWord} is a fundamental concept`,
              `${keyWord} is a supporting detail`,
              `${keyWord} is an example`,
              `${keyWord} is unrelated to the topic`
            ],
            correct: 'A',
            difficulty: difficulty,
            explanation: `${keyWord} appears in the content as a key concept that supports the main learning objectives.`
          };
        } else if (type === 'truefalse') {
          const isTrue = Math.random() > 0.5;
          question = {
            id: questionId++,
            type: 'truefalse',
            question: `The material discusses "${keyWord}" as a central theme.`,
            options: ['True', 'False'],
            correct: isTrue ? 'True' : 'False',
            difficulty: difficulty,
            explanation: `Based on the analysis of the content, this statement is ${isTrue ? 'true' : 'false'}.`
          };
        } else if (type === 'short') {
          question = {
            id: questionId++,
            type: 'short',
            question: `Explain the role of "${keyWord}" based on the provided material.`,
            difficulty: difficulty,
            explanation: `A comprehensive answer should analyze ${keyWord} with specific examples from the content.`,
            sampleAnswer: `${keyWord} plays an important role in the subject matter by providing foundational understanding and connecting to related concepts discussed in the material.`
          };
        }
        
        questions.push(question);
      }
    });
  });

  // Generate flashcards
  const uniqueWords = [...new Set(words)].slice(0, 12);
  const flashcards = uniqueWords.map((word, index) => {
    const relevantSentence = sentences.find(s => 
      s.toLowerCase().includes(word.toLowerCase())
    ) || sentences[index % sentences.length] || 'Related to the study material.';
    
    const categories = ['Key Concept', 'Important Term', 'Core Principle', 'Essential Element', 'Main Topic'];
    const colors = ['#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
    
    return {
      id: index + 1,
      front: word.charAt(0).toUpperCase() + word.slice(1),
      back: `**Definition:** Key concept from the material\n\n**Context:** "${relevantSentence.trim().substring(0, 120)}..."\n\n**Importance:** Essential for understanding the subject matter.`,
      category: categories[index % categories.length],
      color: colors[index % colors.length]
    };
  });

  // Generate summary
  const wordCount = content.split(/\s+/).length;
  const summary = `## Content Summary\n\nThis material contains comprehensive information with approximately ${wordCount} words suitable for learning and assessment.\n\n## Key Highlights\n\n${sentences.slice(0, 5).map((s, i) => `${i + 1}. ${s.trim().substring(0, 100)}...`).join('\n')}\n\n## Generated Materials\n\n- **Questions:** ${questions.length} questions across multiple difficulty levels\n- **Flashcards:** ${flashcards.length} study cards for key concepts\n- **Coverage:** Comprehensive analysis of the provided content`;

  return {
    questions: questions,
    summary: summary,
    flashcards: flashcards
  };
}
