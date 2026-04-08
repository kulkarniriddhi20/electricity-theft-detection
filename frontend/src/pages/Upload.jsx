import axios from "axios";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

export default function Upload() {

  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handlePredict = async () => {
  if (!file) return alert("Upload file first!");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await axios.post("http://localhost:5000/predict", formData);
    navigate("/analytics", { state: res.data });
  } catch (err) {
    alert("❌ Backend not running or error occurred");
    console.error(err);
  }
};
  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-gray-900 text-white flex flex-col items-center justify-center">

      <motion.div 
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        className="bg-white/10 p-10 rounded-2xl backdrop-blur-lg shadow-lg"
      >
        <h1 className="text-2xl mb-5">📤 Upload Customer Data</h1>

        <input 
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          className="mb-4"
        />

        <button
          onClick={handlePredict}
          className="bg-green-500 px-6 py-2 rounded-xl hover:scale-105 transition"
        >
          {loading ? "Processing..." : "🚀 Predict"}
        </button>
      </motion.div>
    </div>
  );
}