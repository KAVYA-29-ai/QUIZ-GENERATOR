# 📘 QuizGen Pro

An AI-powered quiz generator that converts your PDFs or notes into **MCQs, True/False, Short Answer Questions, Summaries, and Flashcards**.  
Built with **React + Vite + Netlify Functions + Google Gemini API**.

---

## 🚀 Features
- 📂 Upload **PDFs or Text files** or paste content directly  
- 🤖 Generate **MCQs, True/False, Short Answer Questions** with explanations  
- 📝 Automatic **content summary**  
- 🃏 Interactive **flashcards** (flip to reveal answers)  
- 🎯 Quiz mode with **dynamic difficulty** and instant feedback  
- 📊 Results dashboard with **score, percentage & performance feedback**  
- ⚡ Serverless backend with **Netlify Functions** + **Gemini API**

---

## 🗂️ Project Structure
quizgen-pro/
│── index.html # React + Babel frontend
│── netlify/
│ └── functions/
│ └── generate.js # Netlify serverless backend using Gemini
│── package.json
│── netlify.toml
│── vite.config.js
│── README.md
│── public/ # Static assets

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/quizgen-pro.git
cd quizgen-pro
2️⃣ Install Dependencies
bash
Copy code
npm install
3️⃣ Configure Environment
Create a .env file in the root folder:

env
Copy code
GEMINI_API_KEY=your_google_gemini_api_key
4️⃣ Run Locally
bash
Copy code
npm run dev
Frontend will be available at http://localhost:5173

5️⃣ Deploy to Netlify
Push repo to GitHub

Connect repo to Netlify

Add environment variable GEMINI_API_KEY in Netlify Project Settings → Environment Variables

Netlify auto-build will deploy both frontend and serverless function 🎉

📸 Screenshots
Upload screen

Generated questions & flashcards

Quiz & results dashboard

🛠️ Tech Stack
Frontend: React (via Babel in index.html), Tailwind-style custom CSS

Backend: Netlify Functions (Node.js)

AI Model: Google Gemini 1.5 Flash

Build Tool: Vite

👨‍💻 Author
Built with ❤️ by Kavya
Feel free to ⭐ the repo and contribute!
