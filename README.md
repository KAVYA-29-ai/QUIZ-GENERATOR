# 🧠 QuizGen Pro

> Turn study material into interactive quizzes, summaries, and flashcards with AI.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-black?style=for-the-badge&logo=netlify)](https://smart-studycreater.netlify.app/)

QuizGen Pro transforms PDFs, text notes, or pasted content into **MCQs, True/False questions, Short Answer questions, summaries, and flashcards** using the Google Gemini API.

## ✨ What It Does

- 📄 Upload PDFs or text files, or paste content directly
- 🤖 Generate questions with AI-powered explanations
- 🧠 Create MCQs, True/False, and Short Answer questions
- 📝 Generate concise study summaries
- 🃏 Review concepts with interactive flashcards
- 🎯 Practice in quiz mode with instant feedback
- 📊 View scores, percentages, and performance insights

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, HTML, CSS |
| AI | Google Gemini API |
| Backend | Netlify Functions / Node.js |
| Deployment | Netlify |

## 🏗️ Architecture

```text
Study Material
     ↓
React + Vite Frontend
     ↓
Netlify Serverless Function
     ↓
Google Gemini API
     ↓
Questions / Summary / Flashcards
     ↓
Interactive Learning Experience
```

## 📁 Project Structure

```text
QUIZ-GENERATOR/
├── netlify/
│   └── functions/
│       └── generate.js
├── public/
├── index.html
├── netlify.toml
├── package.json
├── vite.config.js
└── README.md
```

## 🚀 Run Locally

```bash
npm install
npm run dev
```

Create a `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

For Netlify deployment, add the same environment variable in the site's environment settings.

## 🎨 Design Direction

The interface follows a clean, focused learning experience inspired by Apple's design principles: responsive interactions, clear visual hierarchy, restrained motion, readable typography, and purposeful feedback.

## 🔗 Live Demo

**https://smart-studycreater.netlify.app/**

## 👨‍💻 Author

Built with ❤️ by **Kavya**

⭐ Star the repository if you find it useful.
