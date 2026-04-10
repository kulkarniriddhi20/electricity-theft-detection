import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="p-10 bg-gradient-to-br from-gray-900 to-black min-h-screen text-white">

      <h1 className="text-3xl mb-10 text-center">Dashboard</h1>

      <div className="dashboard-grid">

        {/* Upload Card */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          onClick={() => navigate("/upload")}
          className="p-6 bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg cursor-pointer"
        >
          <h2 className="text-xl mb-2">📂 Upload Data</h2>
          <p>Upload customer CSV file for prediction</p>
        </motion.div>

        {/* Analytics Card */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          onClick={() => navigate("/customer-analytics")}
          className="p-6 bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg cursor-pointer"
        >
          <h2 className="text-xl mb-2">📊 View Analytics</h2>
          <p>View customer-wise analysis and graphs</p>
        </motion.div>

        {/* Future Feature */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          onClick={() => navigate("/alerts")}
          className="p-6 bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg cursor-pointer"
        >
          <h2 className="text-xl mb-2">⚡ Alerts</h2>
          <p>View high-risk electricity theft cases</p>
        </motion.div>

      </div>
    </div>
  );
}