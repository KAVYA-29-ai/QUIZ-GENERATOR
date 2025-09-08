// netlify/functions/generate.js
const { GoogleGenerativeAI } = require('@google/generative-ai');

exports.handler = async (event, context) => {
  // CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS'
  };

  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    const { content, type, config } = JSON.parse(event.body);
    
    if (!content) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Content is required' })
      };
    }

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    let prompt = '';
    
    if (type === 'questions') {
      const { difficultyDistribution, questionTypes } = config;
      const totalQuestions = Object.values(difficultyDistribution).reduce((a, b) => a + b);
      
      prompt = `Based on the following content, generate exactly ${totalQuestions} educational questions with this distribution:

Difficulty Distribution:
- Easy: ${difficultyDistribution.easy} questions
- Medium: ${difficultyDistribution.medium} questions  
- Hard: ${difficultyDistribution.hard} questions
- Expert: ${difficultyDistribution.expert} questions

Question Types:
- Multiple Choice: ${questionTypes.mcq} questions
- True/False: ${questionTypes.truefalse} questions
- Short Answer: ${questionTypes.short} questions

Content:
${content.substring(0, 4000)}

Return ONLY a valid JSON object in this format:
{
  "questions": [
    {
      "id": 1,
      "type": "mcq",
      "difficulty": "easy",
      "question": "Question text",
      "options": ["A", "B", "C", "D"],
      "correct": "A",
      "explanation": "Why this is correct"
    },
    {
      "id": 2,
      "type": "truefalse", 
      "difficulty": "medium",
      "question": "Statement to evaluate",
      "options": ["True", "False"],
      "correct": "True",
      "explanation": "Explanation"
    },
    {
      "id": 3,
      "type": "short",
      "difficulty": "hard", 
      "question": "Open ended question",
      "explanation": "What should be covered",
      "sampleAnswer": "Expected response"
    }
  ]
}`;

    } else if (type === 'flashcards') {
      prompt = `Analyze this content and create 15 educational flashcards. Extract key concepts, terms, and important information.

Content:
${content.substring(0, 4000)}

Return ONLY a valid JSON object:
{
  "flashcards": [
    {
      "id": 1,
      "front": "Key Term/Concept",
      "back": "**Definition:** Clear definition\\n\\n**Context:** How it relates to content\\n\\n**Importance:** Why it matters\\n\\n**Usage:** How it's applied",
      "category": "Primary Concept",
      "color": "#4f46e5"
    }
  ]
}

Use these categories cyclically: "Primary Concept", "Supporting Detail", "Technical Term", "Key Process", "Important Factor"
Use these colors cyclically: "#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"`;

    } else if (type === 'summary') {
      prompt = `Create a comprehensive summary of this content:

${content.substring(0, 4000)}

Return ONLY a valid JSON object:
{
  "summary": "## Content Overview\\n\\nDetailed summary here...\\n\\n## Key Points\\n\\n1. Point one\\n2. Point two\\n\\n## Analysis\\n\\nInsightful analysis..."
}`;

    } else {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Invalid generation type' })
      };
    }

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    // Extract JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No valid JSON found in response');
    }
    
    const jsonData = JSON.parse(jsonMatch[0]);
    
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ success: true, data: jsonData })
    };

  } catch (error) {
    console.error('Error:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ 
        error: 'Generation failed', 
        message: error.message 
      })
    };
  }
};
