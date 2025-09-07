"use client"
import { motion } from "framer-motion"
import { useState } from "react"

export default function App() {
  const [loading, setLoading] = useState(false)

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-900 via-slate-900 to-black p-6">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className="w-full max-w-2xl shadow-2xl rounded-2xl backdrop-blur bg-white/10 border border-white/20 p-6 space-y-4">
          <motion.h1
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.6 }}
            className="text-4xl font-extrabold text-center text-white mb-4"
          >
            QuizGen ✨
          </motion.h1>

          <motion.input
            type="file"
            className="w-full px-3 py-2 bg-white/20 text-white rounded-lg border border-white/30"
            whileFocus={{ scale: 1.02 }}
          />

          <input
            placeholder="Focus topic (optional)"
            className="w-full px-3 py-2 bg-white/20 text-white rounded-lg border border-white/30"
          />

          <select className="w-full px-3 py-2 bg-white/20 text-white rounded-lg border border-white/30">
            <option value="mixed">Mixed (MCQ/TF/Short)</option>
            <option value="mcq">MCQ Only</option>
            <option value="tf">True/False Only</option>
            <option value="short">Short Answer Only</option>
          </select>

          <motion.button
            whileTap={{ scale: 0.9 }}
            whileHover={{ scale: 1.05 }}
            onClick={() => setLoading(true)}
            className="w-full py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold shadow-lg hover:shadow-blue-500/50 transition-all"
          >
            {loading ? "Generating..." : "Generate"}
          </motion.button>
        </div>
      </motion.div>
    </div>
  )
}
