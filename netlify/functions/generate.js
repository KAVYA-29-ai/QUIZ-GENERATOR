// Replace the setTimeout part with actual API call
const generateQuiz = async () => {
  if (!content.trim() && files.length === 0) {
    alert('Please add content or upload files first!');
    return;
  }
  
  setLoading(true);
  
  try {
    const response = await fetch('/.netlify/functions/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content: content,
        type: 'questions',
        config: {
          difficultyDistribution,
          questionTypes
        }
      })
    });
    
    const data = await response.json();
    
    if (data.success) {
      setQuestions(data.data.questions);
      // Also generate summary and flashcards
      await generateSummary(content);
      await generateFlashcards(content);
      setActiveTab('content');
    } else {
      alert('Failed to generate quiz: ' + data.message);
    }
  } catch (error) {
    alert('Error generating quiz: ' + error.message);
  } finally {
    setLoading(false);
  }
};

// Similarly update generateSummary and generateFlashcards functions
const generateSummary = async (text) => {
  try {
    const response = await fetch('/.netlify/functions/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content: text,
        type: 'summary'
      })
    });
    
    const data = await response.json();
    
    if (data.success) {
      setSummary(data.data.summary);
    }
  } catch (error) {
    console.error('Error generating summary:', error);
    // Fallback to basic summary
    const fallbackSummary = `## Content Overview\n\nSummary generation failed. Using basic summary.\n\nContent length: ${text.length} characters`;
    setSummary(fallbackSummary);
  }
};
