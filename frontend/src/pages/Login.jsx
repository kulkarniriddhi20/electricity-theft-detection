import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useState } from "react";

export default function Login() {
  const navigate = useNavigate();
  const [name, setName] = useState(""); // <-- added name
  const [password, setPassword] = useState("");

  const handleLogin = (e) => {
    e.preventDefault();

    if (!name || !password) {
      alert("Please enter name and password");
      return;
    }

    // Show success message with entered name
    alert(`${name} logged in successfully!`);

    // Navigate to Upload page
    navigate("/upload", { state: { username: name } });
  };

  return (
    <div className="h-screen flex justify-center items-center bg-gray-900">
      <motion.form
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        onSubmit={handleLogin}
        className="bg-white p-8 rounded-2xl shadow-lg w-96"
      >
        <h2 className="text-2xl mb-4 text-center text-blue-700">Login</h2>

        {/* Name input */}
        <input
          className="border p-2 w-full mb-3 rounded"
          placeholder="Enter your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        {/* Password input */}
        <input
          type="password"
          className="border p-2 w-full mb-3 rounded"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button
          type="submit"
          className="bg-blue-500 text-white w-full py-2 rounded hover:bg-blue-600 transition"
        >
          Login
        </button>
      </motion.form>
    </div>
  );
}